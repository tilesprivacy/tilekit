<!-- LOGO -->
<p align="center">
  <a href="https://github.com/tileshq/">
    <img src="https://avatars.githubusercontent.com/u/210493283?s=400&u=1ee6e44b6a683b16bdb6e9e853c7ebd8c7fd4268&v=4" alt="Tiles Logo" width="128" />
  </a>
</p>

<h1 align="center">Tiles</h1>

<p align="center">
  Your private AI assistant with offline memory<br />
  <a href="#about">About</a> ·
  <a href="https://tiles.run/download">Download</a> ·
  <a href="https://tiles.run/book">Tiles Book</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="HACKING.md">Developing</a>
</p>

There are a few ways to work with Tiles:

## Tiles CLI
The CLI is the easiest way to start using Tiles. The experience will be familiar to anyone who has worked with e.g. Ollama or LM Studio CLI. You can power it with Claude, GPT, or any other LLM. Just download Tiles, and get started with ``tiles run`` command, that runs the recommended default memory model for your device.

## Tilekit SDK
The SDK is a Modelfile based Rust library that contains all of our model deployment tech. It's the engine that powers everything else in Tiles.

Define models using a Modelfile, the blueprint for creating and sharing models. Our implementation is optimized for fast, efficient local deployment across consumer platforms, starting with Apple devices.

## Tiles GUI
> To be released Q1 26.

Use the Local GUI for chats on your laptop. It includes a REST API and an integration for OpenWebUI. The experience will feel familiar to anyone who has used chatbots like ChatGPT or Claude.

## Download

See the [download page](https://tiles.run/download) on the Tiles website.

## Documentation

See the [Tiles book](https://tiles.run/book) on the Tiles website.

## About

Our mission is to shape the future of software personalization with decentralized memory networks.

Tiles Privacy was born from the [User & Agents](https://userandagents.com) community with a simple idea: software should understand you without taking anything from you. We strive to deliver the best privacy-focused engineering while also offering unmatched convenience in our consumer products. We believe identity and memory belong together, and Tiles gives you a way to own both through your personal user agent.

Tiles is built for privacy conscious users who want intelligence without renting their memory to centralized providers. Our first product is an on-device memory management system paired with an SDK that lets developers securely access user memory and create deeply personalized agent experiences.

We are seeking design partners for training workloads that align with our goal of ensuring a verifiable privacy perimeter. If you're interested, please reach out to us at [hello@tiles.run](mailto:hello@tiles.run)

## Contributing and Developing

If you have any ideas, issues, etc. regarding Ghostty, or would like to
contribute to Tiles through pull requests, please check out our
["Contributing to Tiles"](CONTRIBUTING.md) document. Those who would like
to get involved with Tiles's development as well should also read the
["Developing Tiles"](HACKING.md) document for more technical details.


## License

This project is dual-licensed under MIT and Apache 2.0 terms:

- MIT license [LICENSE-MIT.txt](https://github.com/tileshq/tilekit/blob/main/LICENSE-MIT.txt)
- Apache License, Version 2.0, [LICENSE-APACHE.txt](https://github.com/tileshq/tilekit/blob/main/LICENSE-APACHE.txt)

Downstream projects and end users may choose either license individually, or both together, at their discretion. The motivation for this dual-licensing is the additional software patent assurance provided by Apache 2.0.
