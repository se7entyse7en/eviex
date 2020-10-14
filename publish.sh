set -e
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "master" ]]; then
  echo 'Can bump only from "master" branch';
  exit 1;
fi

git push origin --tags
poetry publish
