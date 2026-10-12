# Deploy (trilomorph.org)

Every push to `main` runs the tests and, if green, builds and pushes
`ghcr.io/cc2872/trilobite_generator:latest` (`.github/workflows/deploy.yml`).
**Watchtower** on the box pulls that image within ~2 minutes and restarts the app, so a
push reaches the live site on its own — no manual step.

## First-time set-up on the box

```sh
git clone https://github.com/cc2872/trilobite_generator    # or just copy docker-compose.yml + .env
cd trilobite_generator
cp .env.example .env         # then fill in the token / record settings
docker compose up -d         # starts app (:8765) + watchtower
```

The tunnel keeps pointing at port 8765. Saved heads/bodies and the recordings persist under
`./data/` (bind-mounted), so they survive both restarts and image updates.

### If the ghcr package is private

Watchtower needs a token to pull. Create a classic PAT with `read:packages`, then either:

- put a Docker config.json next to the compose file and uncomment the `watchtower-config.json`
  mount in `docker-compose.yml`:
  ```json
  { "auths": { "ghcr.io": { "auth": "<base64 of USER:TOKEN>" } } }
  ```
- or `docker login ghcr.io` on the box and mount `~/.docker/config.json` instead.

(If the package is public, nothing extra is needed.)

## Checking a deploy landed

```sh
docker compose logs -f watchtower     # shows when it pulls a new image
docker compose ps                     # app should show a recent "Up"
```

## Manual deploy (if Watchtower is off)

```sh
docker compose pull app && docker compose up -d app
```
