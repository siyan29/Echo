# Echo

A Streamlit app that recognizes a person's face and shows your relation with that person.

## Features

- Add a person with:
  - Name
  - Relation (e.g., friend, sister, colleague)
  - 2-3 reference images for better recognition
- Two recognition modes:
  - **Camera mode** (opens camera and captures an image)
  - **Manual mode** (upload an image)
- Shows recognized name + relation on the detected face.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- For best performance, use clear front-facing images.
- At least 2 uploaded enrollment images must contain a detectable face.
- Face recognition relies on the `face_recognition` package (dlib backend), which may require build tools on some systems.
