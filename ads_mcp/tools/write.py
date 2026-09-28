# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write tools for the MCP server (mutate, Customer Match, click conversions).

Every tool in this module can change a live Google Ads account, so all of them
share the same safety rules:

* ``validate_only`` defaults to ``True``: Google validates the request and
  nothing is changed.
* To really apply a change the caller must pass ``validate_only=False`` AND
  ``confirm=True``. Otherwise the tool refuses.
* Setting the environment variable ``GOOGLE_ADS_MCP_DISABLE_WRITES=true`` turns
  every tool in this module off without redeploying code.
* Personal data (emails, phones, names) is hashed locally with SHA-256 before
  it is sent, and is never written to the logs.
"""

import hashlib
import os
import re
from typing import Any, Dict, List, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from google.ads.googleads.errors import GoogleAdsException
from google.protobuf import json_format
from mcp.types import ToolAnnotations

import ads_mcp.utils as utils
from ads_mcp.mcp_header_interceptor import MCPHeaderInterceptor

write_mcp = FastMCP("write")

_WRITE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False
)

# Limits, kept below the API maximums on purpose.
_MAX_MUTATE_OPERATIONS = 1000
_CUSTOMER_MATCH_CHUNK = 10000
_CONVERSIONS_CHUNK = 1000

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CONVERSION_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$"
)

ConsentStatus = Literal["GRANTED", "DENIED"]


# --------------------------------------------------------------------------
# Safety and plumbing
# --------------------------------------------------------------------------


def _writes_disabled() -> bool:
    value = os.environ.get("GOOGLE_ADS_MCP_DISABLE_WRITES", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _guard(validate_only: bool, confirm: bool) -> None:
    """Raises ToolError unless the requested write is allowed."""
    if _writes_disabled():
        raise ToolError(
            "Write tools are disabled on this server "
            "(GOOGLE_ADS_MCP_DISABLE_WRITES is set)."
        )
    if not validate_only and not confirm:
        raise ToolError(
            "Refusing to apply real changes. Call again with "
            "validate_only=false AND confirm=true, but only after the user "
            "has explicitly approved this exact request."
        )


def _get_client(login_customer_id: str | int | None = None):
    return utils.get_googleads_client(login_customer_id=login_customer_id)


def _get_service(client, service_name: str):
    return client.get_service(service_name, interceptors=[MCPHeaderInterceptor()])


def _raise_api_error(ex: GoogleAdsException) -> None:
    messages = [
        f"Google Ads API Error: {error.message}" for error in ex.failure.errors
    ]
    raise ToolError(f"Request ID: {ex.request_id}\n" + "\n".join(messages))


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _consent_value(client, value: str):
    return getattr(client.enums.ConsentStatusEnum, value)


# --------------------------------------------------------------------------
# Normalization and hashing (Google's Customer Match rules)
# --------------------------------------------------------------------------


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_email(email: Any) -> str | None:
    """Normalizes and hashes an email. Returns None if it is not valid."""
    if not isinstance(email, str):
        return None
    value = email.strip().lower()
    if _HEX64.match(value):
        return value  # already hashed
    if value.count("@") != 1:
        return None
    local, domain = value.split("@")
    if not local or "." not in domain:
        return None
    if domain in ("gmail.com", "googlemail.com"):
        local = local.replace(".", "")
    return _sha256(f"{local}@{domain}")


def hash_phone(phone: Any, default_calling_code: str | None = None) -> str | None:
    """Normalizes to E.164 and hashes a phone. Returns None if not valid."""
    if not isinstance(phone, str):
        return None
    value = phone.strip().lower()
    if _HEX64.match(value):
        return value  # already hashed
    digits = re.sub(r"\D", "", value)
    if value.startswith("+"):
        e164 = "+" + digits
    elif value.startswith("00") and len(digits) > 2:
        e164 = "+" + digits[2:]
    elif default_calling_code:
        code = re.sub(r"\D", "", default_calling_code)
        e164 = "+" + code + digits.lstrip("0")
    else:
        return None
    if not 9 <= len(e164) - 1 <= 15:
        return None
    return _sha256(e164)


def _hash_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip().lower()
    if _HEX64.match(cleaned):
        return cleaned
    return _sha256(cleaned)


def _build_identifiers(
    client, member: Dict[str, Any], default_calling_code: str | None
) -> tuple[list, list[str]]:
    """Builds UserIdentifier objects for one person; returns (ids, problems)."""
    identifiers = []
    problems: list[str] = []

    if member.get("email"):
        hashed = hash_email(member["email"])
        if hashed:
            ident = client.get_type("UserIdentifier")
            ident.hashed_email = hashed
            identifiers.append(ident)
        else:
            problems.append("invalid email")

    if member.get("phone"):
        hashed = hash_phone(member["phone"], default_calling_code)
        if hashed:
            ident = client.get_type("UserIdentifier")
            ident.hashed_phone_number = hashed
            identifiers.append(ident)
        else:
            problems.append("invalid phone (use +country code or set default)")

    address_parts = ("first_name", "last_name", "country_code", "postal_code")
    if any(member.get(k) for k in address_parts):
        first = _hash_text(member.get("first_name"))
        last = _hash_text(member.get("last_name"))
        country = str(member.get("country_code") or "").strip().upper()
        postal = str(member.get("postal_code") or "").strip()
        if first and last and len(country) == 2 and postal:
            ident = client.get_type("UserIdentifier")
            ident.address_info.hashed_first_name = first
            ident.address_info.hashed_last_name = last
            ident.address_info.country_code = country
            ident.address_info.postal_code = postal
            identifiers.append(ident)
        else:
            problems.append(
                "address needs first_name, last_name, 2-letter country_code "
                "and postal_code together"
            )

    if not identifiers and not problems:
        problems.append("no email, phone or address")
    return identifiers, problems


# --------------------------------------------------------------------------
# Tool 1: generic mutate (campaigns, budgets, ads, keywords, audiences...)
# --------------------------------------------------------------------------


@write_mcp.tool(annotations=_WRITE_ANNOTATIONS)
def mutate(
    customer_id: str | int,
    operations: List[Dict[str, Any]],
    validate_only: bool = True,
    confirm: bool = False,
    partial_failure: bool = False,
    login_customer_id: str | int | None = None,
) -> Dict[str, Any]:
    """Creates, updates or removes Google Ads resources (GoogleAdsService.Mutate).

    Covers campaigns, budgets, ad groups, ads, keywords, negative keywords,
    assets, conversion actions, user lists and every other resource that the
    API can mutate.

    SAFETY: by default this only VALIDATES (nothing changes). To apply for
    real, call with validate_only=false and confirm=true, and only after the
    user approved the exact operations.

    Args:
        customer_id: The customer to change, digits only (no hyphens).
        operations: List of MutateOperation objects in JSON form, e.g.
            {"campaignOperation": {"update": {"resourceName":
            "customers/123/campaigns/456", "status": "PAUSED"},
            "updateMask": "status"}}
            Use "create" with a resource, "update" plus "updateMask", or
            "remove" with a resource name. Field names may be camelCase or
            snake_case. Enums are strings.
        validate_only: If true (default) Google only validates the request.
        confirm: Must be true, together with validate_only=false, to apply.
        partial_failure: If true, valid operations apply even if others fail.
        login_customer_id: Optional manager (MCC) id for the login header.

    Returns:
        The API response (resource names of the changed resources).
    """
    _guard(validate_only, confirm)

    if not operations:
        raise ToolError("operations must contain at least one operation.")
    if len(operations) > _MAX_MUTATE_OPERATIONS:
        raise ToolError(
            f"Too many operations ({len(operations)}). "
            f"The limit per call is {_MAX_MUTATE_OPERATIONS}."
        )

    client = _get_client(login_customer_id)
    service = _get_service(client, "GoogleAdsService")

    request = client.get_type("MutateGoogleAdsRequest")
    request.customer_id = utils.clean_customer_id(customer_id)
    request.partial_failure = partial_failure
    request.validate_only = validate_only

    for index, operation in enumerate(operations):
        mutate_operation = client.get_type("MutateOperation")
        try:
            json_format.ParseDict(operation, mutate_operation._pb)
        except json_format.ParseError as ex:
            raise ToolError(f"Operation #{index} is not valid: {ex}")
        request.mutate_operations.append(mutate_operation)

    utils.logger.info(
        "ads_mcp.write.mutate customer=%s ops=%d validate_only=%s",
        request.customer_id,
        len(operations),
        validate_only,
    )
    try:
        response = service.mutate(request=request)
    except GoogleAdsException as ex:
        _raise_api_error(ex)

    result = utils.format_output_value(response)
    return {"validate_only": validate_only, "response": result}


# --------------------------------------------------------------------------
# Tool 2: create a Customer Match list
# --------------------------------------------------------------------------


@write_mcp.tool(annotations=_WRITE_ANNOTATIONS)
def create_customer_match_user_list(
    customer_id: str | int,
    name: str,
    description: str = "",
    membership_life_span_days: int = 10000,
    validate_only: bool = True,
    confirm: bool = False,
    login_customer_id: str | int | None = None,
) -> Dict[str, Any]:
    """Creates an empty Customer Match list (contact info, first-party data).

    After creating it, load people into it with upload_customer_match.
    SAFETY: validates only by default; to create it for real use
    validate_only=false and confirm=true after the user approves.

    Args:
        customer_id: The customer that owns the list, digits only.
        name: List name (must be unique in the account).
        description: Optional description.
        membership_life_span_days: How long people stay in the list. 10000
            means no expiration (default).
        validate_only: If true (default) nothing is created.
        confirm: Must be true, together with validate_only=false, to create.
        login_customer_id: Optional manager (MCC) id for the login header.

    Returns:
        The resource name of the new list (empty when only validating).
    """
    _guard(validate_only, confirm)
    if not name.strip():
        raise ToolError("name is required.")

    client = _get_client(login_customer_id)
    service = _get_service(client, "UserListService")

    operation = client.get_type("UserListOperation")
    user_list = operation.create
    user_list.name = name.strip()
    user_list.description = description
    user_list.membership_life_span = membership_life_span_days
    crm = user_list.crm_based_user_list
    crm.upload_key_type = client.enums.CustomerMatchUploadKeyTypeEnum.CONTACT_INFO
    crm.data_source_type = client.enums.UserListCrmDataSourceTypeEnum.FIRST_PARTY

    cid = utils.clean_customer_id(customer_id)
    utils.logger.info(
        "ads_mcp.write.create_user_list customer=%s validate_only=%s",
        cid,
        validate_only,
    )
    try:
        response = service.mutate_user_lists(
            customer_id=cid,
            operations=[operation],
            validate_only=validate_only,
        )
    except GoogleAdsException as ex:
        _raise_api_error(ex)

    names = [r.resource_name for r in response.results]
    return {"validate_only": validate_only, "resource_names": names}


# --------------------------------------------------------------------------
# Tool 3: upload people to a Customer Match list
# --------------------------------------------------------------------------


@write_mcp.tool(annotations=_WRITE_ANNOTATIONS)
def upload_customer_match(
    customer_id: str | int,
    user_list_id: str | int,
    members: List[Dict[str, Any]],
    ad_user_data_consent: ConsentStatus,
    ad_personalization_consent: ConsentStatus,
    action: Literal["add", "remove"] = "add",
    default_country_calling_code: str | None = None,
    validate_only: bool = True,
    confirm: bool = False,
    login_customer_id: str | int | None = None,
) -> Dict[str, Any]:
    """Adds (or removes) people to a Customer Match list.

    Emails, phones and names are normalized and hashed with SHA-256 locally,
    following Google's rules. Values that are already SHA-256 hashes are sent
    as they are.

    CONSENT: pass GRANTED only if the client confirmed that these people gave
    consent (mandatory for the EEA and the UK, e.g. Spain). Never assume it;
    ask the user.

    SAFETY: by default this only checks the data locally and validates the
    job with Google, and nothing is uploaded. To upload for real use
    validate_only=false and confirm=true after the user approved.

    Args:
        customer_id: The customer that owns the list, digits only.
        user_list_id: Numeric id of an existing Customer Match list.
        members: One dict per person. Accepted keys: email, phone,
            first_name, last_name, country_code (2 letters), postal_code.
            Address data only counts if first_name, last_name, country_code
            and postal_code are all present.
        ad_user_data_consent: GRANTED or DENIED (required, no default).
        ad_personalization_consent: GRANTED or DENIED (required, no default).
        action: "add" (default) or "remove" the people from the list.
        default_country_calling_code: Used for phones without "+", e.g. "54"
            for Argentina, "34" for Spain.
        validate_only: If true (default) nothing is uploaded.
        confirm: Must be true, together with validate_only=false, to upload.
        login_customer_id: Optional manager (MCC) id for the login header.

    Returns:
        Counts of valid and invalid rows and, when applied, the job resource
        name. The job runs asynchronously: check its status with the search
        tool on the offline_user_data_job resource.
    """
    _guard(validate_only, confirm)
    if not members:
        raise ToolError("members must contain at least one person.")

    client = _get_client(login_customer_id)
    cid = utils.clean_customer_id(customer_id)
    list_id = utils.clean_customer_id(user_list_id)
    user_list_resource = f"customers/{cid}/userLists/{list_id}"

    operations = []
    invalid: List[Dict[str, Any]] = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            invalid.append({"row": index, "problem": "not an object"})
            continue
        identifiers, problems = _build_identifiers(
            client, member, default_country_calling_code
        )
        if problems:
            invalid.append({"row": index, "problem": "; ".join(problems)})
        if not identifiers:
            continue
        operation = client.get_type("OfflineUserDataJobOperation")
        user_data = operation.create if action == "add" else operation.remove
        for identifier in identifiers:
            user_data.user_identifiers.append(identifier)
        operations.append(operation)

    summary: Dict[str, Any] = {
        "validate_only": validate_only,
        "user_list": user_list_resource,
        "action": action,
        "rows_received": len(members),
        "rows_ready": len(operations),
        "rows_with_problems": len(invalid),
        "problems": invalid[:50],
    }
    if not operations:
        raise ToolError(
            "No valid rows to upload. Problems: "
            + "; ".join(f"row {p['row']}: {p['problem']}" for p in invalid[:10])
        )

    service = _get_service(client, "OfflineUserDataJobService")
    job = client.get_type("OfflineUserDataJob")
    job.type_ = client.enums.OfflineUserDataJobTypeEnum.CUSTOMER_MATCH_USER_LIST
    metadata = job.customer_match_user_list_metadata
    metadata.user_list = user_list_resource
    metadata.consent.ad_user_data = _consent_value(client, ad_user_data_consent)
    metadata.consent.ad_personalization = _consent_value(
        client, ad_personalization_consent
    )

    utils.logger.info(
        "ads_mcp.write.customer_match customer=%s list=%s rows=%d "
        "action=%s validate_only=%s",
        cid,
        list_id,
        len(operations),
        action,
        validate_only,
    )
    try:
        create_response = service.create_offline_user_data_job(
            customer_id=cid, job=job, validate_only=validate_only
        )
        if validate_only:
            summary["message"] = (
                "Validation passed. Nothing was uploaded. To upload, call "
                "again with validate_only=false and confirm=true."
            )
            return summary

        job_resource = create_response.resource_name
        for chunk in _chunks(operations, _CUSTOMER_MATCH_CHUNK):
            service.add_offline_user_data_job_operations(
                request={
                    "resource_name": job_resource,
                    "operations": chunk,
                    "enable_partial_failure": True,
                }
            )
        service.run_offline_user_data_job(resource_name=job_resource)
    except GoogleAdsException as ex:
        _raise_api_error(ex)

    summary["job_resource_name"] = job_resource
    summary["message"] = (
        "Upload started. Google processes it asynchronously (it can take "
        "several hours). Check the job status with the search tool on "
        "offline_user_data_job."
    )
    return summary


# --------------------------------------------------------------------------
# Tool 4: upload offline click conversions (CRM leads -> Google Ads)
# --------------------------------------------------------------------------


@write_mcp.tool(annotations=_WRITE_ANNOTATIONS)
def upload_click_conversions(
    customer_id: str | int,
    conversion_action_id: str | int,
    conversions: List[Dict[str, Any]],
    ad_user_data_consent: ConsentStatus,
    ad_personalization_consent: ConsentStatus,
    default_country_calling_code: str | None = None,
    validate_only: bool = True,
    confirm: bool = False,
    login_customer_id: str | int | None = None,
) -> Dict[str, Any]:
    """Uploads offline conversions (e.g. qualified CRM leads) to Google Ads.

    Works with the click id (gclid, gbraid or wbraid) and/or with the lead's
    email or phone (enhanced conversions for leads), which are hashed locally.
    The conversion action must already exist and be of type "import" from
    clicks (UPLOAD_CLICKS).

    CONSENT: pass GRANTED only if the client confirmed the consent. Never
    assume it; ask the user.

    SAFETY: validates only by default. To upload for real use
    validate_only=false and confirm=true after the user approved.

    Args:
        customer_id: The customer that owns the conversion action.
        conversion_action_id: Numeric id of the conversion action.
        conversions: One dict per conversion. Keys: conversion_date_time
            (required, "YYYY-MM-DD HH:MM:SS+00:00" with time zone offset),
            gclid / gbraid / wbraid, email, phone, conversion_value,
            currency_code (e.g. "EUR"), order_id. Each row needs a click id
            or an email/phone.
        ad_user_data_consent: GRANTED or DENIED (required, no default).
        ad_personalization_consent: GRANTED or DENIED (required, no default).
        default_country_calling_code: Used for phones without "+", e.g. "54".
        validate_only: If true (default) Google validates and records nothing.
        confirm: Must be true, together with validate_only=false, to upload.
        login_customer_id: Optional manager (MCC) id for the login header.

    Returns:
        How many conversions were sent, local problems, and any partial
        failure message returned by Google.
    """
    _guard(validate_only, confirm)
    if not conversions:
        raise ToolError("conversions must contain at least one row.")

    client = _get_client(login_customer_id)
    service = _get_service(client, "ConversionUploadService")
    cid = utils.clean_customer_id(customer_id)
    action_resource = (
        f"customers/{cid}/conversionActions/"
        f"{utils.clean_customer_id(conversion_action_id)}"
    )

    prepared = []
    invalid: List[Dict[str, Any]] = []
    for index, row in enumerate(conversions):
        if not isinstance(row, dict):
            invalid.append({"row": index, "problem": "not an object"})
            continue

        when = str(row.get("conversion_date_time") or "")
        if not _CONVERSION_TIME.match(when):
            invalid.append(
                {
                    "row": index,
                    "problem": "conversion_date_time must look like "
                    "2026-09-28 14:30:00+00:00",
                }
            )
            continue

        conversion = client.get_type("ClickConversion")
        conversion.conversion_action = action_resource
        conversion.conversion_date_time = when
        for click_id in ("gclid", "gbraid", "wbraid"):
            if row.get(click_id):
                setattr(conversion, click_id, str(row[click_id]))
        if row.get("conversion_value") is not None:
            conversion.conversion_value = float(row["conversion_value"])
        if row.get("currency_code"):
            conversion.currency_code = str(row["currency_code"]).upper()
        if row.get("order_id"):
            conversion.order_id = str(row["order_id"])

        identifiers, problems = _build_identifiers(
            client,
            {"email": row.get("email"), "phone": row.get("phone")},
            default_country_calling_code,
        )
        has_click_id = any(row.get(k) for k in ("gclid", "gbraid", "wbraid"))
        if not has_click_id and not identifiers:
            invalid.append(
                {
                    "row": index,
                    "problem": "needs a click id or a valid email/phone",
                }
            )
            continue
        for identifier in identifiers:
            conversion.user_identifiers.append(identifier)
        conversion.consent.ad_user_data = _consent_value(
            client, ad_user_data_consent
        )
        conversion.consent.ad_personalization = _consent_value(
            client, ad_personalization_consent
        )
        prepared.append(conversion)

    summary: Dict[str, Any] = {
        "validate_only": validate_only,
        "conversion_action": action_resource,
        "rows_received": len(conversions),
        "rows_sent": len(prepared),
        "rows_with_problems": len(invalid),
        "problems": invalid[:50],
        "partial_failure_messages": [],
    }
    if not prepared:
        raise ToolError(
            "No valid conversions to upload. Problems: "
            + "; ".join(f"row {p['row']}: {p['problem']}" for p in invalid[:10])
        )

    utils.logger.info(
        "ads_mcp.write.click_conversions customer=%s rows=%d validate_only=%s",
        cid,
        len(prepared),
        validate_only,
    )
    try:
        for chunk in _chunks(prepared, _CONVERSIONS_CHUNK):
            response = service.upload_click_conversions(
                customer_id=cid,
                conversions=chunk,
                partial_failure=True,
                validate_only=validate_only,
            )
            failure = response.partial_failure_error
            if failure and failure.message:
                summary["partial_failure_messages"].append(failure.message)
    except GoogleAdsException as ex:
        _raise_api_error(ex)

    return summary
