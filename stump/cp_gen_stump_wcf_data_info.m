function wcf_data_info=cp_gen_stump_wcf_data_info(feat_data)

issparse_data=issparse(feat_data);


if issparse_data
    %try to make a sparse sort indexes...
    
    [e_num e_dim]=size(feat_data);
    sorted_feat_data=spalloc(e_num, e_dim,nnz(feat_data));
    sort_idxes=sorted_feat_data;
    for dim_idx=1:e_dim
                
        [one_sorted_feat_data, one_sort_idxes] = sort(feat_data(:,dim_idx), 1); 
        one_sort_idxes(one_sorted_feat_data==0)=0;
        sorted_feat_data(:,dim_idx)=sparse(one_sorted_feat_data);
        sort_idxes(:,dim_idx)=sparse(one_sort_idxes);
      
    end
    wcf_data_info.sorted_feat_data=sorted_feat_data;
    wcf_data_info.sort_idxes=sort_idxes;
else
    [wcf_data_info.sorted_feat_data, sort_idxes] = sort(feat_data, 1); 
    wcf_data_info.sort_idxes = int32(sort_idxes);
end

wcf_data_info.issparse_data=issparse_data;
wcf_data_info.feat_data_ext=cat(2, feat_data, ones(size(feat_data,1),1));

end
