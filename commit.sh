mv ./c*/*.html ./htmls/
mv ./*.html ./htmls/
sh -x sed.sh
git add ./*
git status
git commit -m "xx"
git push origin
