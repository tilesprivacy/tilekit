# HACKING.md

This guide will help you set up a reproducible development environment for Tiles. Tiles supports two environments: `prod` (production) and `dev` (development). These instructions assume you are setting up for local development.

## Prerequisites

- [Rust & Cargo](https://www.rust-lang.org/tools/install)
- [`just`](https://github.com/casey/just) (for task management)
- [Python 3.8+](https://www.python.org/downloads/)
- [`uv`](https://docs.astral.sh/uv/) (for fast Python dependency management)
- [Git](https://git-scm.com/)

## Setup Steps

1. **Clone the repository:**

   ```sh
   git clone https://github.com/tilesprivacy/tiles.git
   cd tiles
   ```

2. **Install Rust dependencies:**

   If you're new to Rust, see [Rust Install Guide](https://www.rust-lang.org/tools/install).

   ```sh
   cargo build
   ```

3. **Install project task runner:**

   [`just`](https://github.com/casey/just) provides easy command shortcuts.

   ```sh
   cargo install just      # or use your OS package manager
   ```

4. **Set up the Python server environment:**

   - Make sure [`uv`](https://docs.astral.sh/uv/) is installed:

     ```sh
     pip install uv
     ```

   - Sync Python dependencies:

     ```sh
     cd server
     uv sync
     cd ..
     ```

## Running Tiles (Development)

Open two terminal windows:

1. **Terminal 1: Start the server**

   From the project root:

   ```sh
   just serve
   ```

2. **Terminal 2: Run the Rust CLI**

   From the root directory:

   ```sh
   cargo run --manifest-path tiles/Cargo.toml
   ```

> **Tip:** Refer to the `justfile` for additional common commands and automation. For troubleshooting, see [CONTRIBUTING.md](CONTRIBUTING.md) and open an issue if you need help.

## Additional Resources

- [Tiles Book](https://tiles.run/book)
- [Download Page](https://tiles.run/download)
- [Community & Support](https://go.tiles.run/discord)
