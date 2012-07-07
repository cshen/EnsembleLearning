
feat_data=[];
label_data=[];
data_weight=[];

thread_num=4;

% do cache
wcf_data_info=cp_gen_stump_wcf_data_info(feat_data);


% find weak learner
wcf_info = build_stump_cpp(wcf_data_info, label_data, data_weight, thread_num);

% the weak learner
weak_cf=wcf_info.weak_cf;
