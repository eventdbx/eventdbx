use std::io::{self, Read, Write};

use anyhow::{Context, Result, anyhow, bail};
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use sha2::{Digest, Sha256};
use snow::{TransportState, params::NoiseParams};

const NOISE_PROTOCOL_NAME: &str = "Noise_NNpsk0_25519_ChaChaPoly_SHA256";
const MAX_MESSAGE_LEN: usize = 16 * 1024 * 1024;
const AEAD_TAG_LEN: usize = 16;
const HANDSHAKE_MESSAGE_MAX: usize = 1024;
const NOISE_MAX_MESSAGE_LEN: usize = u16::MAX as usize;
const NOISE_MAX_PAYLOAD_LEN: usize = NOISE_MAX_MESSAGE_LEN - AEAD_TAG_LEN;
const MAX_ENCRYPTED_CHUNK_LEN: usize = NOISE_MAX_MESSAGE_LEN;
const CHUNK_HEADER_LEN: usize = 1;
const MAX_CHUNK_DATA_LEN: usize = NOISE_MAX_PAYLOAD_LEN - CHUNK_HEADER_LEN;
const CHUNK_FLAG_CONTINUE: u8 = 1;
const CHUNK_FLAG_DONE: u8 = 0;

fn derive_psk(token: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(token);
    let digest = hasher.finalize();
    let mut psk = [0u8; 32];
    psk.copy_from_slice(&digest);
    psk
}

pub async fn perform_client_handshake<R, W>(
    reader: &mut R,
    writer: &mut W,
    token: &[u8],
) -> Result<TransportState>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let params: NoiseParams = NOISE_PROTOCOL_NAME
        .parse()
        .context("failed to parse Noise protocol definition")?;
    let psk = derive_psk(token);
    let builder = snow::Builder::new(params).psk(0, &psk);
    let mut state = builder
        .build_initiator()
        .context("failed to build Noise initiator")?;
    let mut buffer = vec![0u8; HANDSHAKE_MESSAGE_MAX];
    let len = state
        .write_message(&[], &mut buffer)
        .context("failed to write Noise handshake message")?;
    send_frame(writer, &buffer[..len]).await?;
    writer
        .flush()
        .await
        .context("failed to flush Noise handshake message")?;

    let message = read_frame(reader).await?;
    let frame = message.ok_or_else(|| anyhow!("peer closed connection during Noise handshake"))?;
    state
        .read_message(&frame, &mut [])
        .context("failed to read Noise handshake response")?;

    state
        .into_transport_mode()
        .context("failed to construct Noise transport state")
}

pub async fn perform_server_handshake<R, W>(
    reader: &mut R,
    writer: &mut W,
    token: &[u8],
) -> Result<TransportState>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let params: NoiseParams = NOISE_PROTOCOL_NAME
        .parse()
        .context("failed to parse Noise protocol definition")?;
    let psk = derive_psk(token);
    let builder = snow::Builder::new(params).psk(0, &psk);
    let mut state = builder
        .build_responder()
        .context("failed to build Noise responder")?;

    let message = read_frame(reader).await?;
    let frame = message.ok_or_else(|| anyhow!("peer closed connection during Noise handshake"))?;
    state
        .read_message(&frame, &mut [])
        .context("failed to process Noise handshake message")?;

    let mut buffer = vec![0u8; HANDSHAKE_MESSAGE_MAX];
    let len = state
        .write_message(&[], &mut buffer)
        .context("failed to write Noise handshake response")?;
    send_frame(writer, &buffer[..len]).await?;
    writer
        .flush()
        .await
        .context("failed to flush Noise handshake response")?;

    state
        .into_transport_mode()
        .context("failed to construct Noise transport state")
}

pub async fn write_encrypted_frame<W>(
    writer: &mut W,
    state: &mut TransportState,
    plaintext: &[u8],
) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    if plaintext.len() > MAX_MESSAGE_LEN {
        bail!(
            "plaintext message exceeds maximum Noise frame length ({} bytes)",
            MAX_MESSAGE_LEN
        );
    }

    let mut offset = 0;
    loop {
        let remaining = plaintext.len().saturating_sub(offset);
        let chunk_len = remaining.min(MAX_CHUNK_DATA_LEN);
        let more = remaining > chunk_len;

        let mut chunk = Vec::with_capacity(CHUNK_HEADER_LEN + chunk_len);
        chunk.push(if more {
            CHUNK_FLAG_CONTINUE
        } else {
            CHUNK_FLAG_DONE
        });
        if chunk_len > 0 {
            chunk.extend_from_slice(&plaintext[offset..offset + chunk_len]);
        }

        let mut buffer = vec![0u8; chunk.len() + AEAD_TAG_LEN];
        let len = state
            .write_message(&chunk, &mut buffer)
            .context("failed to encrypt Noise frame")?;
        send_frame(writer, &buffer[..len]).await?;
        writer
            .flush()
            .await
            .context("failed to flush encrypted Noise frame")?;

        if !more {
            break;
        }
        offset += chunk_len;
    }
    Ok(())
}

pub async fn read_encrypted_frame<R>(
    reader: &mut R,
    state: &mut TransportState,
) -> Result<Option<Vec<u8>>>
where
    R: AsyncRead + Unpin,
{
    let mut message = Vec::new();
    let mut saw_chunk = false;

    loop {
        let frame = match read_frame(reader).await? {
            Some(frame) => frame,
            None if !saw_chunk => return Ok(None),
            None => {
                bail!("peer closed connection while reading encrypted Noise frame");
            }
        };
        if frame.len() > MAX_ENCRYPTED_CHUNK_LEN {
            bail!(
                "encrypted Noise frame exceeds maximum length ({} bytes)",
                MAX_ENCRYPTED_CHUNK_LEN
            );
        }

        let mut buffer = vec![0u8; frame.len()];
        let len = state
            .read_message(&frame, &mut buffer)
            .context("failed to decrypt Noise frame")?;
        buffer.truncate(len);

        if buffer.is_empty() {
            bail!("decrypted Noise frame missing chunk flag");
        }

        let (chunk_flag, chunk_payload) = buffer.split_first().expect("checked buffer non-empty");
        let more = match *chunk_flag {
            CHUNK_FLAG_DONE => false,
            CHUNK_FLAG_CONTINUE => true,
            other => bail!("invalid Noise chunk flag {other}"),
        };

        if message.len() + chunk_payload.len() > MAX_MESSAGE_LEN {
            bail!(
                "decrypted Noise message exceeds maximum length ({} bytes)",
                MAX_MESSAGE_LEN
            );
        }
        message.extend_from_slice(chunk_payload);
        saw_chunk = true;

        if !more {
            return Ok(Some(message));
        }
    }
}

async fn send_frame<W>(writer: &mut W, payload: &[u8]) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    let len = payload.len();
    if len > u32::MAX as usize {
        bail!("frame payload exceeds u32 length");
    }
    let mut header = [0u8; 4];
    header.copy_from_slice(&(len as u32).to_be_bytes());
    writer
        .write_all(&header)
        .await
        .context("failed to write Noise frame header")?;
    writer
        .write_all(payload)
        .await
        .context("failed to write Noise frame payload")?;
    Ok(())
}

async fn read_frame<R>(reader: &mut R) -> Result<Option<Vec<u8>>>
where
    R: AsyncRead + Unpin,
{
    let mut header = [0u8; 4];
    match reader.read_exact(&mut header).await {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => {
            return Err(anyhow!("failed to read Noise frame header: {}", err));
        }
    }
    let len = u32::from_be_bytes(header) as usize;
    if len > MAX_ENCRYPTED_CHUNK_LEN {
        bail!(
            "frame length {} exceeds allowed maximum {}",
            len,
            MAX_ENCRYPTED_CHUNK_LEN
        );
    }
    let mut payload = vec![0u8; len];
    if len > 0 {
        if let Err(err) = reader.read_exact(&mut payload).await {
            if err.kind() == io::ErrorKind::UnexpectedEof {
                return Ok(None);
            }
            return Err(anyhow!("failed to read Noise frame payload: {}", err));
        }
    }
    Ok(Some(payload))
}

pub fn perform_client_handshake_blocking<S>(stream: &mut S, token: &[u8]) -> Result<TransportState>
where
    S: Read + Write,
{
    let params: NoiseParams = NOISE_PROTOCOL_NAME
        .parse()
        .context("failed to parse Noise protocol definition")?;
    let psk = derive_psk(token);
    let builder = snow::Builder::new(params).psk(0, &psk);
    let mut state = builder
        .build_initiator()
        .context("failed to build Noise initiator")?;
    let mut buffer = vec![0u8; HANDSHAKE_MESSAGE_MAX];
    let len = state
        .write_message(&[], &mut buffer)
        .context("failed to write Noise handshake message")?;
    send_frame_blocking(stream, &buffer[..len])?;
    stream.flush()?;

    let frame = read_frame_blocking(stream)?
        .ok_or_else(|| anyhow!("peer closed connection during Noise handshake"))?;
    state
        .read_message(&frame, &mut [])
        .context("failed to read Noise handshake response")?;

    state
        .into_transport_mode()
        .context("failed to construct Noise transport state")
}

pub fn write_encrypted_frame_blocking<W: Write>(
    writer: &mut W,
    state: &mut TransportState,
    plaintext: &[u8],
) -> Result<()> {
    if plaintext.len() > MAX_MESSAGE_LEN {
        bail!(
            "plaintext message exceeds maximum Noise frame length ({} bytes)",
            MAX_MESSAGE_LEN
        );
    }

    let mut offset = 0;
    loop {
        let remaining = plaintext.len().saturating_sub(offset);
        let chunk_len = remaining.min(MAX_CHUNK_DATA_LEN);
        let more = remaining > chunk_len;

        let mut chunk = Vec::with_capacity(CHUNK_HEADER_LEN + chunk_len);
        chunk.push(if more {
            CHUNK_FLAG_CONTINUE
        } else {
            CHUNK_FLAG_DONE
        });
        if chunk_len > 0 {
            chunk.extend_from_slice(&plaintext[offset..offset + chunk_len]);
        }

        let mut buffer = vec![0u8; chunk.len() + AEAD_TAG_LEN];
        let len = state
            .write_message(&chunk, &mut buffer)
            .context("failed to encrypt Noise frame")?;
        send_frame_blocking(writer, &buffer[..len])?;
        writer.flush()?;

        if !more {
            break;
        }
        offset += chunk_len;
    }
    Ok(())
}

pub fn read_encrypted_frame_blocking<R: Read>(
    reader: &mut R,
    state: &mut TransportState,
) -> Result<Option<Vec<u8>>> {
    let mut message = Vec::new();
    let mut saw_chunk = false;

    loop {
        let frame = match read_frame_blocking(reader)? {
            Some(frame) => frame,
            None if !saw_chunk => return Ok(None),
            None => {
                bail!("peer closed connection while reading encrypted Noise frame");
            }
        };
        if frame.len() > MAX_ENCRYPTED_CHUNK_LEN {
            bail!(
                "encrypted Noise frame exceeds maximum length ({} bytes)",
                MAX_ENCRYPTED_CHUNK_LEN
            );
        }
        let mut buffer = vec![0u8; frame.len()];
        let len = state
            .read_message(&frame, &mut buffer)
            .context("failed to decrypt Noise frame")?;
        buffer.truncate(len);

        if buffer.is_empty() {
            bail!("decrypted Noise frame missing chunk flag");
        }

        let (chunk_flag, chunk_payload) = buffer.split_first().expect("checked buffer non-empty");
        let more = match *chunk_flag {
            CHUNK_FLAG_DONE => false,
            CHUNK_FLAG_CONTINUE => true,
            other => bail!("invalid Noise chunk flag {other}"),
        };

        if message.len() + chunk_payload.len() > MAX_MESSAGE_LEN {
            bail!(
                "decrypted Noise message exceeds maximum length ({} bytes)",
                MAX_MESSAGE_LEN
            );
        }
        message.extend_from_slice(chunk_payload);
        saw_chunk = true;

        if !more {
            return Ok(Some(message));
        }
    }
}

fn send_frame_blocking<W: Write>(writer: &mut W, payload: &[u8]) -> Result<()> {
    let len = payload.len();
    if len > u32::MAX as usize {
        bail!("frame payload exceeds u32 length");
    }
    let mut header = [0u8; 4];
    header.copy_from_slice(&(len as u32).to_be_bytes());
    writer
        .write_all(&header)
        .context("failed to write Noise frame header")?;
    writer
        .write_all(payload)
        .context("failed to write Noise frame payload")?;
    Ok(())
}

fn read_frame_blocking<R: Read>(reader: &mut R) -> Result<Option<Vec<u8>>> {
    let mut header = [0u8; 4];
    match reader.read_exact(&mut header) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => {
            return Err(anyhow!("failed to read Noise frame header: {}", err));
        }
    }
    let len = u32::from_be_bytes(header) as usize;
    if len > MAX_ENCRYPTED_CHUNK_LEN {
        bail!(
            "frame length {} exceeds allowed maximum {}",
            len,
            MAX_ENCRYPTED_CHUNK_LEN
        );
    }
    let mut payload = vec![0u8; len];
    if len > 0 {
        if let Err(err) = reader.read_exact(&mut payload) {
            if err.kind() == io::ErrorKind::UnexpectedEof {
                return Ok(None);
            }
            return Err(anyhow!("failed to read Noise frame payload: {}", err));
        }
    }
    Ok(Some(payload))
}
