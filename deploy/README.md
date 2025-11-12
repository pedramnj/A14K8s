# Systemd Unit Files

This directory contains reference systemd user services for keeping the AI4K8s stack running on long-lived hosts (like the HPC environment).

## Services

- `cloudflared.service`: Maintains the Cloudflare tunnel used to expose the web UI. It expects a named tunnel token stored at `%h/.cloudflared/ai4k8s.token` and writes logs to `%h/cloudflared.log`.
- `ai4k8s-web.service`: Keeps the Flask application (`ai_kubernetes_web_app.py`) running from `%h/ai4k8s` with automatic restarts.

## Usage

1. Copy the service files into `~/.config/systemd/user/` for the target user.
2. Run `systemctl --user daemon-reload` to pick up the new units.
3. Enable lingering if services should survive logout: `loginctl enable-linger $USER`.
4. Enable and start the services:
   systemctl --user enable cloudflared.service ai4k8s-web.service
   systemctl --user start cloudflared.service ai4k8s-web.service
5. Check status as needed: `systemctl --user status cloudflared.service`.

Adjust any paths or environment values to match your deployment.
