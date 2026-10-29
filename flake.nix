{
  description = "Declaration-level dependency graphs for Lean 4 packages";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell ({
          packages = [
            pkgs.python3
            pkgs.uv
            pkgs.ruff
            # Lean toolchain *manager*, not a pinned Lean. lean4export must be
            # built against the target project's own lean-toolchain, which is
            # only known at analysis time -- pinning a lean4 derivation here
            # would fix one version the library has to vary across.
            pkgs.elan
          ];

          # Point uv at the Nix interpreter and stop it fetching its own.
          UV_PYTHON = "${pkgs.python3}/bin/python";
          UV_PYTHON_DOWNLOADS = "never";

          shellHook = ''
            echo "lean-dep-graph dev shell"
            echo "  uv sync && uv run pytest"
          '';
        } // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
          # Let prebuilt binary wheels find a loader. Deliberately NOT
          # LD_LIBRARY_PATH: on non-NixOS Linux that forces the host linker to
          # load Nix's glibc and crashes Python. On NixOS this additionally
          # needs programs.nix-ld.enable = true system-side (see README).
          NIX_LD = pkgs.stdenv.cc.bintools.dynamicLinker;
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
            pkgs.openssl
            pkgs.libffi
          ];
        });
      });
    };
}
