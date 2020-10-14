set -e
branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "master" ]]; then
  echo 'Can bump only from "master" branch';
  exit 1;
fi

new_version=$(bumpversion --dry-run --list $1 | grep new_version | cut -c 13-)
current_version=$(bumpversion --dry-run --list $1 | grep current_version | cut -c 17-)
bumpversion $1
sed -i.bck "s/^-$//g" HISTORY.md
sed -i.bck "s/^-[[:space:]]##/##/g" HISTORY.md
rm HISTORY.md.bck
git commit -a -s -m "Bump version: ${current_version} → ${new_version}"
git tag -a "v${new_version}" -m "Version ${new_version}"
