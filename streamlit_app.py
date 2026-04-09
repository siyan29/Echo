from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import face_recognition
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

DATA_DIR = Path("data")
PROFILE_IMAGE_DIR = DATA_DIR / "profiles"
PROFILE_DB_PATH = DATA_DIR / "profiles.json"


@dataclass
class KnownFace:
    person_id: str
    name: str
    relation: str
    encoding: np.ndarray


def ensure_storage() -> None:
    PROFILE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not PROFILE_DB_PATH.exists():
        PROFILE_DB_PATH.write_text("[]", encoding="utf-8")


def load_profiles() -> list[dict]:
    ensure_storage()
    with PROFILE_DB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_profiles(profiles: list[dict]) -> None:
    PROFILE_DB_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_known_faces() -> List[KnownFace]:
    known_faces: list[KnownFace] = []
    for profile in load_profiles():
        person_id = profile["id"]
        for image_path in profile.get("images", []):
            path = Path(image_path)
            if not path.exists():
                continue
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if not encodings:
                continue
            known_faces.append(
                KnownFace(
                    person_id=person_id,
                    name=profile["name"],
                    relation=profile["relation"],
                    encoding=encodings[0],
                )
            )
    return known_faces


def save_profile(name: str, relation: str, uploads: list) -> tuple[bool, str]:
    if len(uploads) < 2 or len(uploads) > 3:
        return False, "Please upload 2 to 3 images for better recognition."

    person_id = str(uuid.uuid4())
    person_dir = PROFILE_IMAGE_DIR / person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    saved_images: list[str] = []
    valid_face_count = 0

    for index, upload in enumerate(uploads, start=1):
        image = Image.open(upload).convert("RGB")
        image_np = np.array(image)
        encodings = face_recognition.face_encodings(image_np)
        if not encodings:
            continue

        valid_face_count += 1
        path = person_dir / f"face_{index}.jpg"
        image.save(path)
        saved_images.append(str(path))

    if valid_face_count < 2:
        return False, "At least 2 uploaded images must contain a clear face."

    profiles = load_profiles()
    profiles.append(
        {
            "id": person_id,
            "name": name,
            "relation": relation,
            "images": saved_images,
        }
    )
    save_profiles(profiles)
    load_known_faces.clear()
    return True, f"Saved profile for {name} ({relation})."


def identify_faces(image: Image.Image, known_faces: list[KnownFace]) -> tuple[Image.Image, list[str]]:
    rgb_np = np.array(image.convert("RGB"))
    locations = face_recognition.face_locations(rgb_np)
    encodings = face_recognition.face_encodings(rgb_np, locations)

    draw = ImageDraw.Draw(image)
    messages: list[str] = []

    for location, encoding in zip(locations, encodings):
        top, right, bottom, left = location
        label = "Unknown"

        if known_faces:
            distances = face_recognition.face_distance(
                [entry.encoding for entry in known_faces],
                encoding,
            )
            best_match_index = int(np.argmin(distances))
            if distances[best_match_index] < 0.5:
                best = known_faces[best_match_index]
                label = f"{best.name} ({best.relation})"
                messages.append(f"Matched: {best.name} — Relation: {best.relation}")
            else:
                messages.append("Face detected but not recognized in your saved profiles.")
        else:
            messages.append("No saved people yet. Add profiles first.")

        draw.rectangle((left, top, right, bottom), outline="lime", width=3)
        draw.rectangle((left, bottom - 25, right, bottom), fill="lime")
        draw.text((left + 6, bottom - 20), label, fill="black")

    if not locations:
        messages.append("No face detected in this image.")

    return image, messages


def enrollment_panel() -> None:
    st.subheader("Add a person")
    with st.form("add_person"):
        name = st.text_input("Person name")
        relation = st.text_input("Relation with you (e.g., Brother, Friend, Colleague)")
        uploads = st.file_uploader(
            "Upload 2 to 3 face images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("Save person")

    if submitted:
        if not name.strip() or not relation.strip():
            st.error("Please enter both name and relation.")
            return
        ok, message = save_profile(name.strip(), relation.strip(), uploads or [])
        if ok:
            st.success(message)
        else:
            st.error(message)


def show_saved_people() -> None:
    st.subheader("Saved people")
    profiles = load_profiles()
    if not profiles:
        st.info("No people saved yet.")
        return
    for profile in profiles:
        st.write(f"• **{profile['name']}** — {profile['relation']} ({len(profile.get('images', []))} image(s))")


def recognition_panel() -> None:
    st.subheader("Recognize person")
    mode = st.radio("Recognition mode", ["Camera mode", "Manual mode"], horizontal=True)
    known_faces = load_known_faces()

    if mode == "Camera mode":
        st.caption("Open your camera and capture a photo to identify faces.")
        captured = st.camera_input("Capture image")
        if captured:
            image = Image.open(captured).convert("RGB")
            output, messages = identify_faces(image, known_faces)
            st.image(output, caption="Recognition result", use_container_width=True)
            for msg in messages:
                st.write(msg)
    else:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="manual_upload")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            output, messages = identify_faces(image, known_faces)
            st.image(output, caption="Recognition result", use_container_width=True)
            for msg in messages:
                st.write(msg)


def main() -> None:
    st.set_page_config(page_title="Echo Face Relation Recognizer", page_icon="👤", layout="centered")
    st.title("Echo: Face + Relation Recognizer")
    st.write(
        "Save people with their relation, then identify who is in front of the camera "
        "or from an uploaded image."
    )

    enrollment_panel()
    show_saved_people()
    st.divider()
    recognition_panel()


if __name__ == "__main__":
    main()
