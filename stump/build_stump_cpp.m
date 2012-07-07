function wcf_info = build_stump_cpp(wcf_data_info, label_data, data_w_org, thread_num)

sorted_data=wcf_data_info.sorted_feat_data;
sorted_index = wcf_data_info.sort_idxes;


% ??? why should transform in this way ???
% if treat all prediction as pos, the sum_w1 is the -score, the wcf try to
% find the stump with min(-score)
data_w=-label_data.*data_w_org;
sum_w1=sum(data_w);

assert(~issparse(sorted_data));
assert(~issparse(sorted_index));
assert(~issparse(data_w));



% [w_elemt, b, nz_idx, werr] = LibbLearnerSTUMP(sorted_data, data_w, sorted_index, sum_w1);

if thread_num<=1
    [w_elemt, b, nz_idx, werr] = build_stump(sorted_data, data_w, sorted_index, sum_w1);
else
    [w_elemt, b, nz_idx, werr] = build_stump_mt(sorted_data, data_w, sorted_index, sum_w1);
end


if nz_idx<=0
    error('weak classifier failed');
end

W = sparse(1, size(sorted_data, 2));
W(nz_idx) = w_elemt;
B = b;

weak_cf = [W B];

wcf_info.weak_cf=weak_cf;
wcf_info.werr=werr;

scores=sign(wcf_data_info.feat_data_ext*weak_cf');
wcf_info.werr=sum((scores~=label_data).*data_w_org)/sum(data_w_org);


end