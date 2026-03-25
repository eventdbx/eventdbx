# syntax=docker/dockerfile:1.7

ARG BUILDER_IMAGE=rust:1.90-slim-bookworm@sha256:64232e656c058f4468e8d024e990acff04f0fd5a5c0a88a574dc37773d7325c9
ARG DEBUG_IMAGE=debian:stable-slim
ARG RUNTIME_PACKAGES="ca-certificates tzdata libstdc++6 libsnappy1v5 liblz4-1 libzstd1 zlib1g libbz2-1.0 xz-utils"
ARG BUILD_PACKAGES="build-essential clang cmake pkg-config libsnappy-dev liblz4-dev libzstd-dev libbz2-dev zlib1g-dev libssl-dev capnproto"
ARG APP_USER=eventdbx
ARG APP_GROUP=eventdbx
ARG APP_UID=65532
ARG APP_GID=65532

FROM ${BUILDER_IMAGE} as builder

ARG BUILD_PACKAGES
ARG RUNTIME_PACKAGES
ARG APP_USER
ARG APP_GROUP
ARG APP_UID
ARG APP_GID

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    $BUILD_PACKAGES \
    $RUNTIME_PACKAGES \
 && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock build.rs ./
COPY proto ./proto

RUN cargo fetch --locked

COPY src ./src

RUN cargo build --release --locked

RUN find target/release -maxdepth 1 -type f -executable -print -exec strip {} \; || true

RUN set -eux; \
    runtime_root=/runtime-root; \
    install -d "${runtime_root}/usr/local/bin" \
      "${runtime_root}/etc" \
      "${runtime_root}/etc/ssl/certs" \
      "${runtime_root}/usr/share" \
      "${runtime_root}/var/lib/eventdbx/.eventdbx/logs" \
      "${runtime_root}/var/log/eventdbx" \
      "${runtime_root}/tmp"; \
    install -m 0750 /app/target/release/dbx "${runtime_root}/usr/local/bin/dbx"; \
    ldd /app/target/release/dbx \
      | awk '/=> \// { print $3 } /^\// { print $1 }' \
      | sort -u \
      | xargs -r -I '{}' cp --parents -L '{}' "${runtime_root}"; \
    readelf -l /app/target/release/dbx \
      | awk '/Requesting program interpreter/ { sub(/^.*: /, "", $0); sub(/]$/, "", $0); print }' \
      | xargs -r -I '{}' cp --parents -L '{}' "${runtime_root}"; \
    find /lib /usr/lib \( -name 'libnss_dns.so.2' -o -name 'libnss_files.so.2' \) -print \
      | sort -u \
      | xargs -r -I '{}' cp --parents -L '{}' "${runtime_root}"; \
    install -m 0644 /etc/ssl/certs/ca-certificates.crt "${runtime_root}/etc/ssl/certs/ca-certificates.crt"; \
    cp -a /usr/share/zoneinfo "${runtime_root}/usr/share/"; \
    printf 'root:x:0:0:root:/root:/usr/sbin/nologin\n%s:x:%s:%s:EventDBX:/var/lib/eventdbx:/usr/sbin/nologin\n' \
      "${APP_USER}" "${APP_UID}" "${APP_GID}" > "${runtime_root}/etc/passwd"; \
    printf 'root:x:0:\n%s:x:%s:\n' \
      "${APP_GROUP}" "${APP_GID}" > "${runtime_root}/etc/group"; \
    printf '%s\n' \
      'passwd: files' \
      'group: files' \
      'hosts: files dns' \
      'networks: files' \
      'services: files' \
      'protocols: files' \
      'rpc: files' \
      'ethers: files' \
      'shadow: files' > "${runtime_root}/etc/nsswitch.conf"; \
    chmod 1777 "${runtime_root}/tmp"; \
    chown -R "${APP_UID}:${APP_GID}" \
      "${runtime_root}/var/lib/eventdbx" \
      "${runtime_root}/var/log/eventdbx"

FROM ${DEBUG_IMAGE} AS debug

ARG RUNTIME_PACKAGES
ARG APP_USER
ARG APP_GROUP
ARG APP_UID
ARG APP_GID

LABEL org.opencontainers.image.source="https://github.com/thachp/eventdbx" \
      org.opencontainers.image.description="EventDBX server and CLI" \
      org.opencontainers.image.licenses="MIT"

ENV EVENTDBX_DATA_DIR=/var/lib/eventdbx
ENV EVENTDBX_AUTO_INIT=1
ENV HOME=${EVENTDBX_DATA_DIR}
ENV PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Secrets are loaded from env vars at runtime; supply them with `docker run -e ...` when needed:
#   EVENTDBX_DATA_ENCRYPTION_KEY – 32-byte base64 string encrypting on-disk data
#   EVENTDBX_AUTH_PRIVATE_KEY – base64 Ed25519 private key used for JWT signing
#   EVENTDBX_AUTH_PUBLIC_KEY – matching Ed25519 public key distributed to verifiers

RUN apt-get update \
 && apt-get install -y --no-install-recommends $RUNTIME_PACKAGES \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder --chown=${APP_UID}:${APP_GID} --chmod=750 /app/target/release/dbx /usr/local/bin/dbx

RUN groupadd --system --gid "${APP_GID}" "${APP_GROUP}" \
 && useradd --system \
    --home "${EVENTDBX_DATA_DIR}" \
    --shell /usr/sbin/nologin \
    --uid "${APP_UID}" \
    --gid "${APP_GID}" \
    "${APP_USER}" \
 && install -d -m 750 -o "${APP_USER}" -g "${APP_GROUP}" \
      "${EVENTDBX_DATA_DIR}" \
      "${EVENTDBX_DATA_DIR}/.eventdbx/logs" \
      /var/log/eventdbx \
 && chown -R "${APP_USER}:${APP_GROUP}" "${EVENTDBX_DATA_DIR}" /var/log/eventdbx

USER ${APP_USER}
WORKDIR ${EVENTDBX_DATA_DIR}
VOLUME ["${EVENTDBX_DATA_DIR}"]

EXPOSE 7070 6363

ENTRYPOINT ["dbx"]
CMD ["serve", "start", "--foreground"]

FROM scratch

LABEL org.opencontainers.image.source="https://github.com/thachp/eventdbx" \
      org.opencontainers.image.description="EventDBX server and CLI" \
      org.opencontainers.image.licenses="MIT"

ENV EVENTDBX_DATA_DIR=/var/lib/eventdbx
ENV EVENTDBX_AUTO_INIT=1
ENV HOME=${EVENTDBX_DATA_DIR}
ENV PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Secrets are loaded from env vars at runtime; supply them with `docker run -e ...` when needed:
#   EVENTDBX_DATA_ENCRYPTION_KEY – 32-byte base64 string encrypting on-disk data
#   EVENTDBX_AUTH_PRIVATE_KEY – base64 Ed25519 private key used for JWT signing
#   EVENTDBX_AUTH_PUBLIC_KEY – matching Ed25519 public key distributed to verifiers

COPY --from=builder /runtime-root /

USER 65532:65532
WORKDIR ${EVENTDBX_DATA_DIR}
VOLUME ["${EVENTDBX_DATA_DIR}"]

EXPOSE 7070 6363

ENTRYPOINT ["dbx"]
CMD ["serve", "start", "--foreground"]
