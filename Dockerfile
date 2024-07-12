FROM ubuntu:latest
LABEL authors="Leo"

ENTRYPOINT ["top", "-b"]