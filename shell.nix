let 
    unstable = import (fetchTarball https://nixos.org/channels/nixos-unstable/nixexprs.tar.xz) { };
in
{ pkgs ? import <nixpkgs> {} }:
(pkgs.mkShell {
  name = "pip-env";
  nativeBuildInputs = with pkgs; [
    python310
    poetry
    isort
    stdenv.cc
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc
    ]};
  '';
})
