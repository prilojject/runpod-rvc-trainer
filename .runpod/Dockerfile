# Hub prefers Dockerfile in .runpod/ over the repo root.
# v1.0.0 and v1.0.1 Hub builds died immediately after clone, at
# "Creating cache directory.", before the root Dockerfile ran. Hub always
# builds from a Dockerfile and will not use a pre-pushed GHCR tag on its own.
# This thin file only pulls the already-tested public image (anonymous pull
# verified) so the Hub builder does not rebuild from nvidia/cuda.
FROM ghcr.io/prilojject/runpod-rvc-trainer:v1.0.0@sha256:4813a66d8388e3af0cfe0cfbb5c646c0bff2bfb543ea0ca544017f8cc2415a30
