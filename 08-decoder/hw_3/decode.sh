. ./path.sh

# steps/decode.sh --nj 5 --cmd "run.pl" exp/tri1/graph_nosp_tgsmall \
#     data/dev_clean_2 exp/tri1/decode_nosp_tgsmall_dev_clean_2

data=exp/tri1/decode_nosp_tgsmall_dev_clean_2/lat.1.gz

lattice-copy --write-compact=false "ark:gunzip -c $data|" ark,t:lat.1.txt
