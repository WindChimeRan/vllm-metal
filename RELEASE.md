# Release process

1. Update `[project].version` in `pyproject.toml` and `.github/vllm-release-tag.commit` on `main`.
2. Cut the branch: `git switch -c releases/v0.4.0 main && git push origin releases/v0.4.0`.
3. Run **Build and Release** from that branch with `0.4.0rc1`.
4. Test representative attention types, including Qwen3 (GQA) and Qwen3.5 (hybrid SDPA + GDN), plus speculative decoding.
5. Run the workflow again with `0.4.0`.
6. For a later fix, cherry-pick it to the branch and release `0.4.0.post1`.
