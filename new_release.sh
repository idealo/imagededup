uv tool install bump-my-version
bump-my-version bump $1
git add .
git commit -m "Bump version to $(bump-my-version show current_version)"
git push

uv build --sdist

# Wait for wheels to be built via GA
# Move downloaded wheels to dist/
# Test wheels locally, if successful:
# uv publish --path dist/

