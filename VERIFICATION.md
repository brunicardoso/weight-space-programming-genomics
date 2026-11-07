# Verification

This release was exported from a private git repository whose history is not published (it contains the answer keys and run archives; see REDACTIONS.md). The table fixes that history: for each milestone, the full commit hash, its date, and the SHA-256 of the key file **as stored in the private repository at that commit** (`git show <commit>:<path> | sha256sum`).

The last column says whether the exported copy is byte-identical to that private file. Files are not identical when placeholder substitution changed them (see README), or when the file changed in later commits. The scrubbing is deterministic, so a reviewer given access to the private repository can rerun `harness/export_public.py` there and compare with MANIFEST.md.

| milestone | commit | date | file | SHA-256 at that commit (private) | exported copy identical |
|---|---|---|---|---|---|
| Preregistration freeze (tag prereg-v1) | `204caf5c5ccba95e190038729ff8143357fb0e9b` | 2026-11-04 12:07:38 -0300 | `reports/preregistration.md` | `f267e632b86143767ea80bdc0ba06afb0037060071a6be18b31b44ceb3489200` | no |
| Amendment 1 text | `d7f49f51e9eae3c08c2e32f51774c7c137a94be2` | 2026-11-05 09:08:32 -0300 | `reports/preregistration.md` | `dd5e8195ebdce34b677629b98bb6d8b8df269716e9041020b19e33e1b7a5f827` | no |
| Amendment 1 code | `b85ef5cd29846f051f8063276ad0bf810a2d34de` | 2026-11-05 09:06:44 -0300 | (code commit) | – | – |
| Amendment 2 text | `0ee62ff704bd74aec4bbc603f2d2c103c472586b` | 2026-11-05 21:26:09 -0300 | `reports/preregistration.md` | `e429ffd777fd9b8cca3ab6b2adc982b8bac6b69f2cba347c13657237cf7d5438` | no |
| Amendment 2 code | `d45480adf59e44eb3544922998719d5867cd42a4` | 2026-11-05 21:15:45 -0300 | (code commit) | – | – |
| Amendment 3 text | `fe1266bee123a6aa4412cc7ad6714d9327d8ce0a` | 2026-11-05 21:35:30 -0300 | `reports/preregistration.md` | `e4e471c1562b7e505bd7b404e64dbcccae9583607055db04bce11779d506815c` | no |
| Amendment 3 code | `7f6e343b03362c7c5cc022dc40f8215c26ed7059` | 2026-11-05 21:33:37 -0300 | (code commit) | – | – |
| Amendment 4 text | `b60ec1cb073de38e93d960b25a245865c7303482` | 2026-11-06 07:49:21 -0300 | `reports/preregistration.md` | `997bcdf2f10cc85915ac0c84c65164904e2d4890991e7248e1b4b1aecc11e2fe` | no |
| Amendment 4 code | `217b918effe537be1c314b348c8a4b4e562bde8e` | 2026-11-06 07:48:48 -0300 | (code commit) | – | – |
| Amendment 5 text | `9f93e9afbf00752b6947b0d3c31fe699ae8ae2bb` | 2026-11-06 09:51:01 -0300 | `reports/preregistration.md` | `57be7cda8142bcac2ddc88e7e12e576e18db8b9c30fc63de8249fbd1d8ebad35` | no |
| Amendment 5 code | `fbb7b7eb73dbf6d9a2073fa759f77826a0d942c0` | 2026-11-06 09:51:01 -0300 | (code commit) | – | – |
| Amendment 6 text | `e490f8558489fb9eadb3a0f508d7dd422fea7b7c` | 2026-11-06 12:28:01 -0300 | `reports/preregistration.md` | `d5eb653a471a8a79c3d108c0a6017513a43a1f4e6ef9d9b72c02e61d238b0d83` | no |
| Amendment 6 text (disclosure) | `bc7855264817d1f14b88244dd78172bb2cff6bcd` | 2026-11-06 12:28:11 -0300 | `reports/preregistration.md` | `75350347d3228b90016a68758828c179500d54dc1f180285b14f9403c4998ed1` | no |
| Amendment 6 code | `e15975188d9fb7b77efafff4cbf7f82fe4147cda` | 2026-11-06 12:25:40 -0300 | (code commit) | – | – |
| Amendment 7 text | `0c1d59fab3899e8f1b70ae9f99ab58358e57c85e` | 2026-11-06 15:00:01 -0300 | `reports/preregistration.md` | `f763f9eeec7e183cd887ea8e07c6e003f07976c0218e0a7f45577af31edde3c7` | no |
| Amendment 7 text (correction) | `9d391c4b18e18be4f43725dfe12a37c0e239006f` | 2026-11-06 15:00:13 -0300 | `reports/preregistration.md` | `fa5cfe094641d4a0d7f732aab6a4fb316e01a5fc09dadda44105fd26d2ab2446` | no |
| Amendment 7 code | `3802b40c439e56cca6e628962f55122f41ba1595` | 2026-11-06 14:56:30 -0300 | (code commit) | – | – |
| Amendment 8 text | `2fb98f9154e1768b95166026cf73c37550d3f461` | 2026-11-06 18:43:33 -0300 | `reports/preregistration.md` | `3ff37270e8d349cecea2854c3ddd1dea19123fff47d0dd0000386154e9a82799` | no |
| Amendment 9 text | `f8ae076ab7589d2f3893a72fc42e07a75d4d9671` | 2026-11-06 20:01:46 -0300 | `reports/preregistration.md` | `607c8141cda281f4d7b8502ca3a2c43be743b41025bfd55e7779a44380f93687` | no |
| Amendment 10 text | `84e4b9d35a59c10218ec955749a2bf7641be2ae7` | 2026-11-06 21:05:13 -0300 | `reports/preregistration.md` | `96949b669b87a401f674efe0766d8ce3a0b90217807d75bcb608e5a93e569f95` | no |
| Amendment 10 erratum | `3943b0b1e35255f51f3fd211791141080c12c339` | 2026-11-06 21:18:05 -0300 | `reports/preregistration.md` | `7c65c2a2f6bb373b36b3a7e002722cb045fbb60a943a27396d2397c7532b2c7d` | no |
| Amendment 11 code | `728a2f064e5e5cf4bfb71730180fc7a609f34944` | 2026-11-06 21:24:39 -0300 | `src/wspg/analysis.py` | `43c658aa1c90b398bbef0ad0c12715db7ff426d5ec2961c41b5b294bcf64d112` | yes |
| Amendment 11 text | `8146b5a0f2bd166b72553d83f3ec04b63c923250` | 2026-11-06 21:25:10 -0300 | `reports/preregistration.md` | `869c0b3bc6bb7d24fb8a6ee2c54f0cac7793c3415e14682f0129df268b26adc0` | no |
| Amendment 12 text | `ced8af4aeea401ac36d3bd57a0934efb2b710701` | 2026-11-06 22:17:56 -0300 | `reports/preregistration.md` | `a7305a93398b82ee83b809124eb095e55b636f4cfd9f075c9a5082b00326fe1b` | no |
| Amendment 13 text | `bd0ff716f36f97dbc905731a30866cfd1d0d9baf` | 2026-11-06 22:29:10 -0300 | `reports/preregistration.md` | `bfb2120cf495770182be93dd029cd8ef690382eb917eccf371d202a84ae46576` | no |
| Pre-unblinding checklist (last commit before scoring) | `5e87fd5150db3056311a87fba2360407bad04a6c` | 2026-11-06 22:43:23 -0300 | `reports/results.md` | `16b57e6c89f9ec644e64b3894617d3153ba485d82cc0bdd9503eb332439ef557` | no |
| Results entry | `3a164a7b5fcb66acae23887917ba3db5b5eac7dc` | 2026-11-07 06:32:54 -0300 | `reports/results.md` | `8b4c497e6a3a0123ade04ee5d44864ab5af832a0996542c715f2836ec3152970` | no |
| Results entry, final disclosure fix | `0d4125b1c076c98c12ffc050de1f9e59b155848f` | 2026-11-07 06:35:12 -0300 | `reports/results.md` | `23d8cdd2f397663b76643de90e5989e9e227c7ddf8e5b33dce8e978f037d2de8` | no |
| Preprint v1 (tag preprint-v1) | `468b0ee8cb88d592d22bd0f883b67ee259b1c6cf` | 2026-11-07 08:51:59 -0300 | `reports/preprint/main.pdf` | `9283e2c16bf9d732edb37f4478ad6776f4cef2ffc0f285da50dcaf9f63e1d26c` | no |
| Preprint before retitling | `f4c3cecd13871c1b1d13958312b54c0b52505273` | 2026-11-07 09:05:12 -0300 | `reports/preprint/main.pdf` | `ff41f679e9ac3b07fe4e305252709fe041bea1d13944bbf308ebc9932afbb987` | no |
| Preprint v1.1, new title (tag preprint-v1.1) | `e7ecfbc531608008f3a6f46cd9f8cf65094b3c58` | 2026-11-07 13:14:45 -0300 | `reports/preprint/main.pdf` | `5c610ee00dd2509132191c7f372ef27890c01df865b81d385c3cbb516232052d` | yes |

Tag objects: `prereg-v1` = `f49946dc3de8b18d290734d37b1b617a702296f4`; `preprint-v1` = `3a0c55dfddfef5202183541560e30afcff19919c`.

The preregistration also publishes the SHA-256 of the sealed run schedule (`sealed/schedule.json`, withheld), recorded before any session ran.

## Recomputing
```bash
sha256sum reports/preregistration.md reports/results.md reports/preprint/main.pdf   # files in this release
# in the private repository:
git show <commit>:<path> | sha256sum
git rev-parse prereg-v1 preprint-v1
```
