# Usage: ./new_release.sh <version>
# Example: ./new_release.sh post for post release, patch for patch release, minor for minor release, major for major release
uv tool install bump-my-version
uv pip install twine
bump-my-version bump $1
git add .
git commit -m "Bump version to $(bump-my-version show current_version)"
git push

uv build --sdist

# Wait for wheels to be built via GA
# Move downloaded wheels to dist/
# Test wheels locally, if successful: twine upload dist/*

