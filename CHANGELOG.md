# Changelog

## v0.1.6 (2026-06-06)

[Compare changes](https://github.com/thememium/dspy-rlm-hooks/compare/v0.1.5...v0.1.6)

### 🏡 Chore

- bump usechange to 0.1.35 ([6217ddd](https://github.com/thememium/dspy-rlm-hooks/commit/6217dddbce0dc5b0f612dc1c91cec02c92abeea3))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.5 (2026-06-06)

[Compare changes](https://github.com/thememium/dspy-rlm-hooks/compare/v0.1.4...v0.1.5)

### 🏡 Chore

- bump usechange to 0.1.35 ([6217ddd](https://github.com/thememium/dspy-rlm-hooks/commit/6217dddbce0dc5b0f612dc1c91cec02c92abeea3))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.4 (2026-05-18)

[Compare changes](https://github.com/thememium/dspy-rlm-hooks/compare/v0.1.3...v0.1.4)

### 📖 Documentation

- **rlm**: replace Mermaid flowchart with ASCII diagram ([b15a553](https://github.com/thememium/dspy-rlm-hooks/commit/b15a5539114a29ddaec58ec9623800df6d1bdc2c))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.3 (2026-05-18)

[Compare changes](https://github.com/thememium/dspy-rlm-hooks/compare/v0.1.2...v0.1.3)

### 🚀 Enhancements

- **workflow**: add publish.yml for CI/CD publishing ([459917f](https://github.com/thememium/dspy-rlm-hooks/commit/459917ff2c93921e57408d0a2ef3c64e40603ff2))

### 🏡 Chore

- **pyproject.toml**: remove MIT license classifier ([916a581](https://github.com/thememium/dspy-rlm-hooks/commit/916a581c3fd786de0c039f355795fcab21e3b751))
- **pyproject**: add descriptive metadata, license, classifiers and package discovery ([51800f0](https://github.com/thememium/dspy-rlm-hooks/commit/51800f02817e798b60916ef4ce63674c6440bba1))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.2 (2026-05-18)

[Compare changes](https://github.com/thememium/dspy-rlm-hooks/compare/v0.1.1...v0.1.2)

### 🩹 Fixes

- **utils**: handle language‑tagged and adjacent code fences ([27c526c](https://github.com/thememium/dspy-rlm-hooks/commit/27c526c438a6a649200f7f4467292b99ff0ef1ff))

### 💅 Refactors

- **conftest**: add dotenv loading ([a90979c](https://github.com/thememium/dspy-rlm-hooks/commit/a90979ce004945eb822f5c992e446d7264445fd4))
- **pyproject.toml**: collapse markers array to single line ([773bdcc](https://github.com/thememium/dspy-rlm-hooks/commit/773bdcc08aeada072ef54fe4c6d01818d15ca460))
- **test**: improve formatting and consistency of test suite ([b098bca](https://github.com/thememium/dspy-rlm-hooks/commit/b098bca96e584292b846f6bf16de76089039d4a4))
- **dspy_rlm_hooks/__init__.py**: add __version__ and reformat imports ([4ac0274](https://github.com/thememium/dspy-rlm-hooks/commit/4ac02746c44f7f3fe466862f806d1e9ced51483a))
- **dspy_rlm_hooks**: add async runner helper and streamline imports ([a45950c](https://github.com/thememium/dspy-rlm-hooks/commit/a45950cc522c1b4517ca7bbd3e250684ee06b05d))

### 📖 Documentation

- add bug report issue template ([68d94f2](https://github.com/thememium/dspy-rlm-hooks/commit/68d94f206086517f66fc31f2cbd5c6dbcbc36acd))
- add contributing guide ([dd7dd7d](https://github.com/thememium/dspy-rlm-hooks/commit/dd7dd7dc0e9a56d7e41fb5d388c7f95734aff32f))
- add SECURITY.md with reporting process and security practices ([05d1594](https://github.com/thememium/dspy-rlm-hooks/commit/05d159407a085d8cd0ab41cca69a3e3c86039bb5))

### 🏡 Chore

- **pyproject**: add filterwarnings to suppress deprecation warnings ([9c1dcf5](https://github.com/thememium/dspy-rlm-hooks/commit/9c1dcf5c400959a42c12cbe4f70be4262a5e01ab))
- **pyproject**: add python-dotenv and e2e test script ([c24e842](https://github.com/thememium/dspy-rlm-hooks/commit/c24e8423070ba969c95da71f6d21316e976e4ee3))
- **pytest**: add pytest‑asyncio and async test configuration ([fdcd5f0](https://github.com/thememium/dspy-rlm-hooks/commit/fdcd5f068c88aa527dfc167361f927ef2d7967dd))
- add MIT license file ([312158a](https://github.com/thememium/dspy-rlm-hooks/commit/312158a2b8337dee6fcfd60e9b8d5740cf295b53))

### ✅ Tests

- improve LLM call exception handling in end‑to‑end tests ([2a5dba9](https://github.com/thememium/dspy-rlm-hooks/commit/2a5dba9d5161495b70aadf9c9a9e9169398e1267))
- add DSPy 3.1+ compatibility tests ([e923ab4](https://github.com/thememium/dspy-rlm-hooks/commit/e923ab42211821a60cabb53be46552c329ed6135))
- add tests for _strip_code_fences utility ([b0deccb](https://github.com/thememium/dspy-rlm-hooks/commit/b0deccb1417aaaced6f06d66001fd73eeeecc545))
- **smoke**: add smoke tests for package import and API presence ([ad20b23](https://github.com/thememium/dspy-rlm-hooks/commit/ad20b23dc1db17a63bda31d37639e7ac3db397cd))
- add comprehensive patcher tests ([b41b33b](https://github.com/thememium/dspy-rlm-hooks/commit/b41b33be4ccc2930b45eb8c71b08c71ff7d3da36))
- **hooks**: add comprehensive lifecycle hook tests ([98fb0cd](https://github.com/thememium/dspy-rlm-hooks/commit/98fb0cdd614c489702fd7886f8c2dcd7daeb4587))
- **e2e**: add comprehensive end‑to‑end tests for RLM hooks ([8d37b73](https://github.com/thememium/dspy-rlm-hooks/commit/8d37b73e1458308541c413ec13f46a85badd4d72))
- add comprehensive tests for enable/disable rlm hooks ([f37b9d9](https://github.com/thememium/dspy-rlm-hooks/commit/f37b9d95f0f72877246bcdcc5bb7c41b97ee9684))
- **test_async**: add async hook tests ([c34e82d](https://github.com/thememium/dspy-rlm-hooks/commit/c34e82d524f5a15b3a925c80879e0dfaf8646350))
- add shared pytest fixtures for dspy-rlm-hooks ([3a48770](https://github.com/thememium/dspy-rlm-hooks/commit/3a48770b0862f185b3b013aa105ba39aaa74a424))

### 🎨 Styles

- **dspy_rlm_hooks**: reformat import statements for consistency ([5b75f48](https://github.com/thememium/dspy-rlm-hooks/commit/5b75f483e96aa7b6fcfa38ab4735aba883e30574))

### Other Changes

- Merge pull request #1 from thememium/feat/add-tests (#1) ([9e5a248](https://github.com/thememium/dspy-rlm-hooks/commit/9e5a248db880885f4b8a96d98a18694440ac4608))

### Contributors

- Edward Boswell <thememium@gmail.com>

## v0.1.1 (2026-05-17)

### 🚀 Enhancements

- **dspy_rlm_hooks**: add lifecycle hooks and public API for RLM ([58bb10d](https://github.com/thememium/dspy-rlm-hooks/commit/58bb10df0beab6429aed8d294c34a4828d61f168))
- **dspys**: add lifecycle hooks for DSPy RLM via monkey‑patch ([bdd12f2](https://github.com/thememium/dspy-rlm-hooks/commit/bdd12f250c4cc5afa75355007485ce896df46d08))
- **dspy_rlm_hooks**: add type definitions for hook system ([863489a](https://github.com/thememium/dspy-rlm-hooks/commit/863489a49517c4dfb0bbb88f96b8e286130815a7))
- **dspy_rlm_hooks**: add _strip_code_fences utility ([38e62e3](https://github.com/thememium/dspy-rlm-hooks/commit/38e62e3786ed2dfbedd14e558e158cbab74491b6))

### 📖 Documentation

- add comprehensive README with usage and development guide ([6c81749](https://github.com/thememium/dspy-rlm-hooks/commit/6c81749c98348523eee94d56d5715515ac9eb799))

### 🏡 Chore

- **deps**: downgrade dspy to 3.1.0 and pydantic to 2.0.0 ([709ceea](https://github.com/thememium/dspy-rlm-hooks/commit/709ceeafdf9658d77ec2b9480f8e837b6eb0dbff))
- **pyproject**: add dspy and pydantic dependencies, remove scripts, bump ty ([39154da](https://github.com/thememium/dspy-rlm-hooks/commit/39154dacc760de02930fef20e4059dc8aa1006a9))

### Contributors

- Edward Boswell <thememium@gmail.com>
