# VS Code tips

The following consist of optional suggestions for configuring VS Code as well as for installing helpful extensions to it. If you use VS Code as your code editor for contributing to this project you may find them useful.

## Settings

An easy way to add settings to VS Code is to declare them in the JSON settings file (F1 → Open User Settings (JSON)). However, if you are unsure about messing with JSON, you can also search for the name of the individual settings in the normal settings UI (F1 → Open User Settings).

```json
"files.autoSave": "afterDelay",
"editor.rulers": [
    120
],
"python.analysis.typeCheckingMode": "basic",
"github.gitProtocol": "ssh",
"python.testing.pytestEnabled": true,
"autoDocstring.docstringFormat": "google-notypes",
"evenBetterToml.schema.enabled": false,
```

!!! info "Layers of settings"
    There are different *layers* of settings. Depending on the scope you want these settings to have, you can apply them at the “User” level, “Remote”, “Workspace” or in a specific folder.

!!! tip "Hierarchy of layers of settings"
    - As mentioned, there are different levels of settings. The hierarchy is as follows: User → Remote → Workspace → Folder. The User settings are on top and apply globally to any instance of VS Code you open.
    
    - Each level down becomes more specific, with its settings overriding those of “higher levels”. So, if you change a value in the Remote settings that also exists in the User settings, the Remote value will take precedence. Of course, this value would only apply to the currently open Remote instance.
    - Similarly with Workspace and Folder. A Workspace is a project you have open. Note that workspace settings are stored at the root of the project in a `.vscode` folder.

!!! info "Settings for extensions"
    Some of these settings apply to optional extensions. If you don't have those extensions, these settings will just be ignored. You can still safely copy the string below into your settings file.

## Extensions

### Core extensions

We list these extensions with their unique identifier (like `ms-python.python`), so that you can easily install them from the command line or add them to the `remote.SSH.defaultExtensions` setting. You can also use that identifier to find the extension in the normal extension view (F1 → Extensions: Install Extensions).

Install via CLI:

```bash
code --install-extension ms-python.python                              # Python support
code --install-extension ms-vscode-remote.vscode-remote-extensionpack  # Remote Development extension pack
code --install-extension GitHub.vscode-pull-request-github             # GitHub Pull Requests and Issues
```

Add to default extensions (F1 → Open User Settings (JSON)):

```json
"remote.SSH.defaultExtensions": [
  "ms-python.python",
  "GitHub.vscode-pull-request-github",
]
```

### Useful extensions

These are not mandatory, but helpful in a lot of cases.

Install via CLI:

```bash
code --install-extension ms-vscode-remote.vscode-remote-extensionpack  # Remote Development extension pack
code --install-extension ms-python.python                   # Python extension
code --install-extension ms-toolsai.jupyter                 # Jupyter extension
code --install-extension charliermarsh.ruff                 # Ruff linter extension
code --install-extension eamodio.gitlens                    # GitLens extension (code insights via git history)
code --install-extension mhutchie.git-graph                 # A graphical git log
code --install-extension rioj7.command-variable             # Needed for the "Current Module" debug profile
code --install-extension ryu1kn.partial-diff                # Create diff of sections within files
code --install-extension njpwerner.autodocstring            # Generate docstrings from function signature
code --install-extension kevinrose.vsc-python-indent        # Helps getting Python indentations right
code --install-extension GitHub.vscode-pull-request-github  # GitHub Pull Requests and Issues
code --install-extension GitHub.copilot                     # GitHub Copilot
code --install-extension GitLab.gitlab-workflow             # GitLab Workflow
code --install-extension mtxr.sqltools                      # Facilitates working with SQL databases
code --install-extension mtxr.sqltools-driver-mysql         # MySQL driver for SQLTools extension
code --install-extension ms-azuretools.vscode-docker        # Docker support for VS Code
code --install-extension tamasfe.even-better-toml           # TOML Language Support
code --install-extension redhat.vscode-xml                  # XML Language Support
code --install-extension redhat.vscode-yaml                 # YAML Language Support
code --install-extension yzhang.markdown-all-in-one         # Markdown Language Support
code --install-extension howcasperwhat.comment-formula      # LaTeX formulas in comments rendered in pop-up
code --install-extension bierner.emojisense                 # Search for emojis
code --install-extension brunnerh.insert-unicode            # Search for Unicode characters
code --install-extension GrapeCity.gc-excelviewer           # Spreadsheet Viewer for csv/tsv/xls files
code --install-extension bierner.markdown-mermaid           # Supports Mermaid diagrams in Markdown preview
code --install-extension ms-vscode.wordcount                # Adds word count to status bar
```

If you use a remote server it is not necessary to do this again in the remote server.
Add to SSH default extensions when using a remote server (F1 → Open User Settings (JSON)):

```json
"remote.SSH.defaultExtensions": [
    "ms-python.python",
    "ms-toolsai.jupyter",
    "charliermarsh.ruff",
    "eamodio.gitlens",
    "mhutchie.git-graph",
    "rioj7.command-variable",
    "ryu1kn.partial-diff",
    "njpwerner.autodocstring",
    "kevinrose.vsc-python-indent",
    "GitHub.vscode-pull-request-github",
    "GitHub.copilot",
    "GitLab.gitlab-workflow",
    "mtxr.sqltools",
    "mtxr.sqltools-driver-mysql",
    "ms-azuretools.vscode-docker",
    "tamasfe.even-better-toml",
    "redhat.vscode-xml",
    "redhat.vscode-yaml",
    "yzhang.markdown-all-in-one",
    "howcasperwhat.comment-formula",
    "bierner.emojisense",
    "brunnerh.insert-unicode",
    "valentjn.vscode-ltex",
    "GrapeCity.gc-excelviewer",
    "bierner.markdown-mermaid",
    "ms-vscode.wordcount"
]
```
