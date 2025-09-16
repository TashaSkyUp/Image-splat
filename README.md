# Image-GS (Worker Pool)

This project packages the `original.md` prototype into a runnable React + Vite application. It implements a browser-only
reference version of Image-GS that trains 2D Gaussian primitives on the CPU while offloading heavy computation to a pool of web
workers.

## Getting started

```bash
npm install
npm run dev
```

Open the printed URL in your browser, load an image, and experiment with the worker-pool training controls.

## Available scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Start the development server with hot module replacement. |
| `npm run build` | Produce a production build in the `dist/` directory. |
| `npm run preview` | Preview the production build locally. |
| `npm run lint` | Run ESLint on the source files. |

## Deployment

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that builds the site with Vite and deploys the
`dist/` output to GitHub Pages via the `actions/deploy-pages` action.
