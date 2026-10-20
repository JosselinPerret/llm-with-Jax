# Deploy this folder to a Hugging Face Gradio Space.
# Prerequisites:
#   1. Create a WRITE token: https://huggingface.co/settings/tokens
#   2. Run: hf auth login
#      (paste the write token when prompted)
#
# Usage (from repo root):
#   .\space\deploy.ps1
#   .\space\deploy.ps1 -SpaceId "almostjojo/tinystories-jax-gpt"

param(
    [string]$SpaceId = "almostjojo/tinystories-jax-gpt"
)

$ErrorActionPreference = "Stop"
$SpaceDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Whoami:" -ForegroundColor Cyan
hf auth whoami

Write-Host "`nCreating Space $SpaceId (Gradio + ZeroGPU free tier)..." -ForegroundColor Cyan
hf repos create $SpaceId `
    --type space `
    --space-sdk gradio `
    --flavor zero-a10g `
    --public `
    --exist-ok

Write-Host "`nUploading files from $SpaceDir ..." -ForegroundColor Cyan
hf upload $SpaceId $SpaceDir . --repo-type space

Write-Host "`nDone. Open: https://huggingface.co/spaces/$SpaceId" -ForegroundColor Green
Write-Host "Build logs: https://huggingface.co/spaces/$SpaceId?logs=build"
