lm=data/local/lm/lm_tglarge.arpa.gz
out_dir=data/lang_nosp_test_tglarge

gunzip -c $lm | arpa2fst --disambig-symbol=#0 --read-symbol-table=$out_dir/words.txt - $out_dir/G.fst

