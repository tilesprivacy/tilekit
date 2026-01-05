There are two environments in Tiles: `prod` and `dev`. Follow the instructions below to set up the development environment.

## Building

1. Clone the repository.

2. Install [`just`](https://github.com/casey/just).

3. Set up the Rust environment:

```sh
cargo build
```

4. Install [`uv`](https://docs.astral.sh/uv/) for the Python server.

5. Set up the server:

```sh
cd server
uv sync
```

## Running

1. From the repository root, start the server in one terminal:

```sh
just serve
```

2. In another terminal, run the Rust CLI using Cargo as usual.

```sh
cd tiles

cargo run
```
