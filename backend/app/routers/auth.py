"""
SAHELI Backend — Authentication endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db import (create_user, get_user_by_email, get_user_by_id, init_db, SUPPORTED_COUNTRIES,
                 update_user_profile, log_activity, get_activity_counts, get_activity_total)
from auth import hash_password, verify_password, create_access_token, decode_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

init_db()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    country: str
    organization: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdateRequest(BaseModel):
    full_name: str | None = None
    organization: str | None = None
    bio: str | None = None
    photo_base64: str | None = None


def _public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "full_name": user["full_name"],
        "country": user["country"],
        "organization": user["organization"],
        "bio": user.get("bio"),
        "photo_base64": user.get("photo_base64"),
        "created_at": user.get("created_at"),
    }


@router.get("/countries")
def list_countries():
    return {"countries": SUPPORTED_COUNTRIES}


@router.post("/register")
def register(payload: RegisterRequest):
    if payload.country not in SUPPORTED_COUNTRIES:
        raise HTTPException(status_code=400, detail=f"Country must be one of: {', '.join(SUPPORTED_COUNTRIES)}")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if get_user_by_email(payload.email):
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    user = create_user(
        email=payload.email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        country=payload.country,
        organization=payload.organization,
    )
    token = create_access_token(user["id"], user["email"])
    return {"token": token, "user": _public_user(user)}


@router.post("/login")
def login(payload: LoginRequest):
    user = get_user_by_email(payload.email)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token(user["id"], user["email"])
    log_activity(user["id"], "login")
    return {"token": token, "user": _public_user(user)}


def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict:
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user_by_id(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_current_user_optional(token: str | None = Depends(oauth2_scheme)) -> dict | None:
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    return get_user_by_id(int(payload["sub"]))


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    return _public_user(user)


@router.patch("/me")
def update_profile(payload: ProfileUpdateRequest, user: dict = Depends(get_current_user)):
    if payload.photo_base64 and len(payload.photo_base64) > 2_000_000:
        raise HTTPException(status_code=400, detail="Photo too large — please use an image under ~1.5MB")
    updated = update_user_profile(
        user["id"], full_name=payload.full_name, organization=payload.organization,
        bio=payload.bio, photo_base64=payload.photo_base64,
    )
    log_activity(user["id"], "profile_update")
    return _public_user(updated)


@router.get("/activity")
def get_my_activity(user: dict = Depends(get_current_user)):
    return {
        "counts": get_activity_counts(user["id"], days=365),
        "total": get_activity_total(user["id"]),
        "member_since": user.get("created_at"),
    }
