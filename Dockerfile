FROM rust:latest

RUN apt-get update && apt-get install -y libfftw3-dev

WORKDIR /FHE

COPY ./FHE /FHE