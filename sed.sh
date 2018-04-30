for i in `ls ./htmls/*.html`
do
    echo "replacing..." $i
    awk '{if($0~/hd101wyy.markdown-preview-enhanced/){print "<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/katex@0.9.0/dist/katex.min.css\" integrity=\"sha384-TEMocfGvRuD1rIAacqrknm5BQZ7W7uWitoih+jMNFXQIbNl16bO8OZmylH/Vi/Ei\" crossorigin=\"anonymous\">" } else {print $0}}' $i > $i.new
    mv $i.new $i
done
