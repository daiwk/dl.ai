mv ./c*/*.html ./htmls/
mv ./*.html ./htmls/
#sh -x sed.sh # no need if generate pdf using cdn hosted
git add ./*
git status
git commit -m "xx"
git push origin
