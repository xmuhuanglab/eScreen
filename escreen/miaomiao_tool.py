import torch
import numpy as np
import pandas as pd
from collections import Counter

def match_summit(data,summit_ref,threshold=200):
    for i in data.index.to_list():
        chromo=data.loc[i,'chr']
        ref_sub=summit_ref[summit_ref['chr']==chromo]
        ref_sub['delta']=abs(ref_sub['start']-data.loc[i,'loci'])
        ref_sub=ref_sub.sort_values(by='delta')
        
        if len(ref_sub)>0:
            if ref_sub.loc[ref_sub.index.to_list()[0],'delta']>threshold:
                data.loc[i,'summit_group']='loss'
                data.loc[i,'summit_value']= 1e5
                data.loc[i,'summit_delta']=ref_sub.loc[ref_sub.index.to_list()[0],'delta']
            else:
                data.loc[i,'summit_group']=ref_sub.loc[ref_sub.index.to_list()[0],'peak_name']
                data.loc[i,'summit_value']=ref_sub.loc[ref_sub.index.to_list()[0],'value']
                data.loc[i,'summit_delta']=ref_sub.loc[ref_sub.index.to_list()[0],'delta']
        else:
            data.loc[i,'summit_group']='loss'
            data.loc[i,'summit_value']= 1e5
            data.loc[i,'summit_delta']= 1e8
            
        print(f'{i}/{len(data)}',end='\r')
    return data

def capture_summit(data, summit_ref, bw_dict, 
                   epi='', sequence_len=200, kmer=40):
    nearest_summits = []
    for chromo in data['chr'].unique():
        summit_subset =summit_ref[chromo]
        if not summit_subset.empty:
            for _, b_row in data[data['chr'] == chromo].iterrows():
                distances = np.abs(summit_subset['start'] - (b_row['start']+b_row['end'])/2 )
                nearest_index,nearest_distance = distances.idxmin(),distances.min()
                if nearest_distance<500:
                    nearest_summits.append((b_row.name, summit_subset.loc[nearest_index]))

    for b_index, nearest in nearest_summits:
        data.loc[b_index, f'{epi}_nearest_summit'] = nearest['value']

    data=data.fillna(1e6)

    summit_array=[]
    for i in data.index.to_list():
        _a_=np.array([ catch_epi( bw_dict[epi],(data.loc[i,'chr'],data.loc[i,'start']+k) )/data.loc[i,f'{epi}_nearest_summit'] \
                       for k in range(int( sequence_len-kmer+1) ) ])[np.newaxis,:,:] 
        summit_array.append(_a_.squeeze())
    data[f'{epi}_summit_weight']=summit_array

    return data

def get_value(data,bw,feature='profile',threshold=0.5,_filter_=True,_verbose_=True):
    
    if data is None:
        pass
    else:
        array_house=[]
        ok_array=[]
        for i in data.index.to_list():
            c=data.loc[i,'chr']
            a,b=min(data.loc[i,'start'],data.loc[i,'end']),max(data.loc[i,'start'],data.loc[i,'end'])
            try:
                array=np.array( bw.values(c,a,b) )
                array[np.isnan(array)] = 0
                array_house.append( array )
                element_counts = Counter(array)
                most_common_element, most_common_count = element_counts.most_common(1)[0]
                ok_array.append( (most_common_count <= (b - a) *threshold) | (most_common_count !=0 ) )
                #ok_array.append( most_common_count <= (b - a) *threshold )
            except:
                array_house.append( np.zeros(200) )
                ok_array.append(1<0)
        data[feature]=array_house
        data['ok']=ok_array
        if _filter_:
            data=data[data['ok']]
        if _verbose_:
            print(len(data[data.y>0]),end="/")
            print(len(data[data.y==0]))
    
    return data

def ln(x): # 自然对数
    return np.log(x)/np.log(np.e)

def lg(x): # 常用对数
    return np.log(x)/np.log(10)

def sigmoid_normalization_point(array,s1=(0.0,0.3),s2=(-0.3,0.7)):
    # 自定义通过任意两点的sigmoid函数
    e=2.71828
    
    x1,y1=s1
    x2,y2=s2
    
    n= ( ln(1/y1-1)-ln(1/y2-1) )/( x2-x1 )
    b= ln(1/y1-1)+n*x1 
    
    print(f'parameter: n={n:.4f},b={b:.4f}')
    
    n_array=1/( 1+e**(n*-1*array + b ) )
    
    return n_array

def nonzero_mean(arr, axis):
    """
    计算输入数组沿指定维度非零元素的平均值。
    
    参数：
    arr (np.ndarray): 输入的数组。
    axis (int): 指定的维度。
    
    返回：
    np.ndarray: 沿指定维度非零元素的平均值。
    """
    # 将数组中零值置为 NaN，这样均值计算时可以忽略
    arr_with_nan = np.where(arr != 0, arr, np.nan)
    
    # 计算非零元素的均值（忽略 NaN）
    mean_result = np.nanmean(arr_with_nan, axis=axis)
    
    return mean_result

def norml(x):
    # 最大-最小值归一化
    if isinstance(x, torch.Tensor):
        x = x.to('cpu')
        x = x.numpy()
    else:
        x=np.array(x)
        
    return (x-x.min())/(x.max()-x.min())

def safe_divide(a, b):
    # 安全地进行除法操作
    a = np.array(a)
    b = np.array(b)
    result = np.zeros_like(a, dtype=np.float64)
    non_zero_mask = b != 0
    result[non_zero_mask] = a[non_zero_mask] / b[non_zero_mask]
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        return pd.Series(result)
    return result

def safe_log(a, base=2):
    # 安全地进行对数操作
    a = np.array(a)  
    result = np.zeros_like(a, dtype=np.float64)
    positive_mask = a > 0
    result[positive_mask] = np.log(a[positive_mask]) / np.log(base)  # 使用换底公式
    if isinstance(a, pd.Series):
        return pd.Series(result)
    return result

def remove_overlap(df1, df2):
    # 用来去除两个数据框中重叠的部分
    overlap = pd.merge(df1, df2, how='inner') # 使用 merge 方法找到重叠部分
    df1_non_overlap = df1[~df1.apply(tuple, 1).isin(overlap.apply(tuple, 1))] # 去除 df1 中与 df2 重叠的部分
    df2_non_overlap = df2[~df2.apply(tuple, 1).isin(overlap.apply(tuple, 1))] # 去除 df2 中与 df1 重叠的部分
    return df1_non_overlap, df2_non_overlap


def power_norm(data, max_val=10, gamma=0.424):
    # 在两个方向上分别进行归一化操作
    abs_data = np.abs(data)
    scaled = (abs_data / max_val) ** gamma
    return np.clip(scaled, 0, 1)


