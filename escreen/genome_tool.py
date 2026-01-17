from pybedtools import BedTool
from pyliftover import LiftOver
import pyBigWig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

base_to_array={'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
chr_list=[f'chr{i}' for i in range(1,23)]+['chrX']
trans_dict={'A':'T','C':'G','G':'C','T':'A','N':'N'}

def reverse_seq(seq):
    # 把DNA序列翻转成它的互补链
    return ''.join([trans_dict[b] for b in seq])

def one_hot(seq):
    # 简简单单的独热编码
    return np.stack( np.array( [base_to_array[a] for a in seq] ) )

def catch_epi(bw=None,loci=[]):
    # 安全地返回某个位点上的表观信息
    if bw==None:
        print('No match tasks.')
        return np.zeros(1)
    
    if len(loci)<1:
        print('No data input.')
        return np.zeros(1)
    
    try:
        values = bw.values( loci[0],int(loci[1]),int(loci[2]) )
    except:
        # 一般这个时候报的错误是边界错误。所以只要解决越界错误就行
        bound = int(bw[loci[0]])
        if int(loci[1])<0 and int(loci[2])<bound: # 起点越界(小于0),不过很难做到起点越界吧
            values = np.concatenate([
                np.zeros(abs(int(loci[1]))),np.array( bw.values( loci[0],0,int(loci[2]) ) )
            ])
        elif int(loci[1])>=0 and int(loci[2])>bound: # 终点越界,最为常见
            values = np.concatenate([
                np.array( bw.values( loci[0],int(loci[1]),bound ) ),np.zeros(int(loci[2])-bound)
            ])
        elif int(loci[1])<0 and int(loci[2])>bound: # 双越界,几乎不可能出现
            values = np.concatenate([
                np.zeros(abs(int(loci[1]))),np.array( bw.values( loci[0],0,bound ) ),np.zeros(int(loci[2])-bound)
            ])            
    return values

def genome_sequence(data,fa,chr_list=chr_list):
    #抓取基因组序列,返回独热编码后的结果
    data=data[data['chr'].isin(chr_list)]
    seq_input=np.stack([ 
        one_hot( 
            fa[ data.loc[i,'chr']][ data.loc[i,'start']:data.loc[i,'end'] ].seq.upper()
        ) for i in tqdm(data.index.to_list())
    ])
    return seq_input

def _genome_sequence_withoutcoding_(data,fa,chr_list=chr_list):
    #抓取基因组序列,但是返回的是序列而不是独热编码后的结果
    data=data[data['chr'].isin(chr_list)]
    seq_input=[ fa[ data.loc[i,'chr'] ][ data.loc[i,'start']:data.loc[i,'end'] ].seq for i in data.index.to_list()]
    return seq_input

def genome_sequence_withoutcoding(data,fa,chr_list=chr_list):
    #抓取基因组序列,但是返回的是序列而不是独热编码后的结果
    data=data[data['chr'].isin(chr_list)]
    seq_input=[ fa[ data.loc[i,'chr'] ][ data.loc[i,'start']:data.loc[i,'end'] ].seq for i in rainbow_tqdm(data.index.to_list())]
    return seq_input

def loci_split(df,length=200):
    # 把大的基因组区间拆分成指定长度的小基因组区间
    result = []
    for _, row in df.iterrows():
        chrom = row['chr']
        start = row['start']
        end = row['end']

        while start < end:
            if end-start>length:
                segment_end = min(start + length, end)
                result.append({'chr': chrom, 'start': start, 'end': segment_end})
                start = segment_end
            else:
                segment_end = max(start + length, end)
                result.append({'chr': chrom, 'start': start, 'end': segment_end})
                start = segment_end
    split_df = pd.DataFrame(result)
    print(len(split_df))
    return split_df

def merge_intervals(df):
    # 把一个文件中的区间尽可能合并(不处理数值,简单合并成一个大区间)
    df=df.reset_index(drop=True)
    
    for i in range(len(df)):
          
        if df.loc[i,'start']>df.loc[i,'end']:
            s,e=df.loc[i,'start'],df.loc[i,'end']
            df.loc[i,'start'],df.loc[i,'end']=e,s

        if df.loc[i,'end']-df.loc[i,'start']<3:
            df.loc[i,'end']=df.loc[i,'start']+20
    
    df[['chr','start','end']].to_csv('intervals.bed',sep='\t',index=False,header=None)
    a = BedTool('intervals.bed').sort()
    merged = a.merge()
    merged.saveas('merged_intervals.bed')
    merged_df=pd.read_csv('merged_intervals.bed',sep='\t',header=None,names=['chr','start','end'])
    
    os.remove('intervals.bed')
    os.remove('merged_intervals.bed')
    
    return merged_df

def convert_genome(region,lo=None):
    # 把基因组坐标在不同的基因组之间转换（同一物种）
    if not lo is None:#需要输入一个转换参考
        a,b,c=region[1],region[2],region[0]
        d=b-a#序列长度
        status=0#工作状态
        count = 0
        while (status<1):
            converted = lo.convert_coordinate(c, a, b)        
            if not converted is None:#保证转换出来不是空的
                if len(converted)>0:
                    chrom,s = converted[0][0],converted[0][1]
                    status=1#终止
                else:
                    a+=1;b+=1;count+=1
                    if count > 100: 
                        print('Loci can not be converted:',end=' ');print(region)
                        return '0',0,0
                    
            else:
                a+=1;b+=1;count+=1
                if count > 100: 
                    print('Loci can not be converted:',end=' ');print(region)
                    return '0',0,0
                
            if status==1:#正常结束工作
                return chrom,s,s+d
            
            
def ref_filter(r, d):
    # 取两个数据框中基因组位置有重叠的记录，返回的是两个数据框之间相互有重叠的记录
    df_list = []
    outref_list = []
    d_grouped = {chr_: df for chr_, df in d.groupby('chr')}

    for chromo, s, e in zip(r['chr'], r['start'], r['end']):
        if chromo in d_grouped:
            d_chr = d_grouped[chromo]  # 先筛选出染色体相同的部分
            # 找到 start 或 end 落入 [s, e] 范围的行
            mask = (
                ((d_chr['start'].astype(int) >= s) & (d_chr['start'].astype(int) <= e)) |
                ((d_chr['end'].astype(int) >= s) & (d_chr['end'].astype(int) <= e)) |
                ((d_chr['start'].astype(int) <= s) & (d_chr['end'].astype(int) >= e))
            )
            ds = d_chr[mask]
            if not ds.empty:
                df_list.append(ds)
                outref_list.append(r.loc[r['chr'] == chromo])  # 记录匹配到的参考区间

    df = pd.concat(df_list) if df_list else pd.DataFrame()
    outref = pd.concat(outref_list) if outref_list else pd.DataFrame()

    if df.empty:
        print('No activity guide!')

    return outref,df 

def split_coordinates(df,col='Coordinates'):
    # 把chrN:start-end:*格式编码的位置记录拆分成原始的染色体，起点和终点
    df[col] = df[col].str.split('|').str[-1]
    split_df = df[col].str.split(':', expand=True)
    df['chr'] = split_df[0]
    region_split = split_df[1].str.split('-', expand=True)
    df['start'] = region_split[0].fillna(0).astype(int)
    df['end'] = region_split[1].fillna(0).astype(int)
    try:
        df['strand'] = split_df[2].fillna('*')
    except:
        print('All data loss strand info. Use * fill')
        df['strand'] = '*'
    
    return df[['chr', 'start', 'end', 'strand']]


def find_contextual(data,fa=None,expand_flank=30,upstream=4,downstream=6,total_len=30):
    
    # 自动寻找上下文并且把补全上下文之后的序列整理成有30bp的格式
    ## 因为参考基因组并不会随着细胞系的改变而改变，所以这里可以这么做

    direction=[];seq_adj=[];position_adj=[]
    
    merge_table=data
    
    for _,row in tqdm(merge_table.iterrows(),total=len(merge_table)):
    
        seq = row['sequence'].upper().replace(' ','')[-20:]

        try:
            c,s,e = row['chr'],int(row['start']),int(row['end'])
            s,e = min(s,e),max(s,e)
            if not c in chr_list:
                direction.append('-1')
                seq_adj.append('N'*total_len)
                position_adj.append(-1)
                continue
        except:
            direction.append('-1')
            seq_adj.append('N'*upstream+s1+'N'*downstream)
            position_adj.append(-1)
            continue

        sequence=fa[c][s-expand_flank:e+expand_flank].seq.upper()

        l1=sequence.find(seq)

        if l1<0: # 没有抓取到位置信息的情况,自动认为是负向的 ## 注意,这个版本取互补链的时候不会把3’和5’倒转过来
            l1=sequence.find(reverse_seq(seq)[::-1])
            if l1<0: # 如果仍然没有抓取到位置信息就使用原始位置且默认为正向
                direction.append('+');seq_adj.append( fa[c][s-upstream:s+20+downstream].seq.upper() )
                position_adj.append(s)
            else: # 如果抓取到位置信息就记为负向
                direction.append('-');seq_adj.append( reverse_seq( fa[c][s-expand_flank+l1-downstream:s-expand_flank+l1+20+upstream].seq.upper() )[::-1] )
                position_adj.append(s-expand_flank+l1)
        else:
            direction.append('+');seq_adj.append( fa[c][s-expand_flank+l1-upstream:s-expand_flank+l1+20+downstream].seq.upper() )
            position_adj.append(s-expand_flank+l1)

    # direction是找到的guide的方向
    # seq_adj是调整之后的guide
    # position_adj是调整之后补全的guide的起点位置
    
    return direction,seq_adj,position_adj


    
def eqtlboxplot_draw(data,chromo,lo=None,label='eQTL boxplot figure',label_list=[],guide_expand=False,expand=90,sample_limit=200):
    # 根据eqtl文件绘制直方图
    min_start=min( [data[i].start.min() for i in range(len(data)) ] )
    max_end=max( [data[i].end.max() for i in range(len(data)) ] )
    
    eqtlx=pd.read_csv("/cluster2/huanglab/liquan/data/eQTL/ciseqtl-loci.txt",sep='\t')
    
    import concurrent.futures

    def process_group(ii,guide_expand=guide_expand,expand=expand):
        groupresults = []
        intervals = list(ii[['chr','start','end']].itertuples(index=False, name=None))
        results = {}
        i = 0

        for chrom, start, end in intervals:
            
            if chrom=='chrX':
                groupresults.append(np.nan)
                continue
            
            if guide_expand:
                start=start-expand
                end=end+expand
            
            if not lo is None:
                converted = lo.convert_coordinate(chrom, start, end)
                chrom = converted[0][0]
                s, e = converted[0][1], converted[0][3]
                if abs(s - e) > abs(start - end) * 1.1:
                    if s > e:
                        delta = e - end
                        start = start + delta
                        end = e
                    else:
                        delta = s - start
                        end = end + delta
                        start = s
                else:
                    start, end = s, e


            if start == end:
                groupresults.append(0)
            else:
                start, end = min(start, end), max(start, end)
                print(start, end)
                try:
                    groupresults.append(eqtlx[(eqtlx.SNPChr==int(chrom.split('chr')[1])) & (eqtlx.SNPPos >= start) & (eqtlx.SNPPos <= end)].shape[0])
                except:
                    print(f'bad region: {chrom}-{start}-{end}')
                    groupresults.append(0)

            i += 1
            print(f'{i}/{len(ii)}\r', end="")


        ii['snp'] = groupresults
        print('')
        return ii

    # 定义你想要处理的组列表
    groups = data
    
    # 使用 ThreadPoolExecutor 进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_group, groups))

    # 处理结果
    for i,(j,k) in enumerate( zip(data,results) ):
        j=k
        j['label']=label_list[i]
    
    # 将数据框合并为一个
    combined_df = pd.concat(data).reset_index(drop=True)

    #内部比较
    t1, p1 = ttest_ind(data[0]['snp'], data[1]['snp'])
    
    # 绘制箱线图
    plt.figure(figsize=(4, 3.2))
    sns.set_style("ticks")

    sns.boxplot(x='label', y='snp', data=combined_df,
                palette=['#ecf0f1'], showfliers=False, width=0.4)
    sns.stripplot(x='label', y='snp', data=combined_df,
                  jitter=True, alpha=0.4, color='#74b9ff', size=2)

    sf=1.2
    figuretop=max(data[0].snp.max(),data[1].snp.max())+2
    plt.ylim(-1,figuretop)

    x1, x2 = 0, 1 
    y, h, col = figuretop - 1.0*sf, 0.1, 'k'
    if data[0]['snp'].mean()>data[1]['snp'].mean():
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='#eb2f06')
    else:
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='#0984e3')
    plt.text((x1 + x2) * .5, y + h, f'p = {p1:.3g}', ha='center', va='bottom', color=col,fontsize=7)

    plt.title('eQTL enrichment cross groups')
    plt.xticks(fontsize=6,rotation=30)
    plt.ylabel('eQTL loci count')
    
    if label:
        plt.xlabel(label)
    else:
        plt.xlabel('Group')
        label=time.time()
        
    # 获取当前轴
    ax = plt.gca()
    # 设置 y 轴的内刻度
    # ax.set_yticks([0,1,2,3]) 
    ax.tick_params(axis='y', which='both', direction='in', length=4)    
    ax.tick_params(axis='x', which='both', direction='out', length=4)   
    
    plt.tight_layout()    
    #plt.savefig(f'figure/eQTLboxplot/{label}.png')    
    plt.show()
    
    #combined_df.to_csv(f'./temp/eqtlboxplot_{label}.csv', index = False)
    return None
    
    
def gwashistplot_draw(data,chromo,gwas_data,lo=None,label='GWAS figure',label_list=[],guide_expand=False,expand=90):
    # 根据gwas文件绘制直方图                                                    
    import concurrent.futures
    
    def process_group(ii,guide_expand=guide_expand,expand=expand):
        groupresults = []
        intervals = list(ii[['chr','start','end']].itertuples(index=False, name=None))
        results = {}
        i = 0
        for chrom, start, end in intervals:
            if chrom=='chrX':
                groupresults.append(np.nan)
                continue      
            if guide_expand:
                start=start-expand
                end=end+expand      
            status=0
            if not lo is None:
                while (status<1):
                    converted = lo.convert_coordinate(chrom, start, end)        
                    if not converted is None:
                        if len(converted)>0:
                            chrom = converted[0][0]
                            s, e = converted[0][1], converted[0][3]
                            status=1
                        else:
                            start+=1
                            end+=1
                    else:
                        groupresults.append(0)
                        status=2
                        break
                if status==2:
                    continue                  
                if abs(s - e) > abs(start - end) * 1.1:
                    if s > e:
                        delta = e - end
                        start = start + delta
                        end = e
                    else:
                        delta = s - start
                        end = end + delta
                        start = s
                else:
                    start, end = s, e
            if start == end:
                groupresults.append(0)
            else:
                start, end = min(start, end), max(start, end)
                try:
                    #conditions=(gwas_data.chromosome==int(chrom.split('chr')[1])) & (gwas_data.base_pair_location >= start) & (gwas_data.base_pair_location <= end)
                    #sub_gwas=gwas_data[conditions]
                    
                    chrom_num = int(chrom.split('chr')[1])
                    sub_gwas = gwas_data.query("chromosome == @chrom_num and base_pair_location >= @start and base_pair_location <= @end")
                    groupresults.append(sub_gwas.shape[0])
                except:
                    #print(f'bad region: {chrom}-{start}-{end}')
                    groupresults.append(0)
            i += 1
            print(f'{i}/{len(ii)}\r', end="")
        ii['gwas'] = groupresults
        print('')
        return ii

    groups = data
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_group, groups))
    
    for i,(j,k) in enumerate( zip(data,results) ):
        j=k
        j['label']=label_list[i]
    
    combined_df = pd.concat([data]).reset_index(drop=True)
    
    plt.figure(figsize=(4, 3.2))
    
    groups = [d['label'].unique()[0] for d in data ] 
    counts = [d['gwas'].sum() for d in data ]
    positions = range(len(groups))
    
    plt.bar(positions, counts, color='skyblue', edgecolor='black', width=0.5)
    plt.xticks(positions, groups,fontsize=6)
    
    #plt.ylim([max(counts)-0.5,max(counts)+0.5])
    
    plt.title('Sum of GWAS loci cross group')
    plt.xlabel(label)
    plt.ylabel('Count')
    plt.tight_layout()
    #plt.savefig(f'figure/GWASbarplot/{label}.png')    
    plt.show()
    print(counts)
    
    
def draw_loci_context(loci):
    ### 给定一个loci自动绘制这个loci上面的上下文图案
    chrchr=loci[0];startstart=loci[1];endend=loci[2]

    bw=pyBigWig.open('/cluster2/huanglab/liquan/data/K562/DNase.bigWig')
    dnase_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()

    bw=pyBigWig.open('/cluster2/huanglab/liquan/data/K562/ATAC.bigWig')
    atac_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()

    bw=pyBigWig.open('/cluster2/huanglab/liquan/data/K562/H3K27ac.bigWig')
    k27ac_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()

    bw=pyBigWig.open('/cluster2/huanglab/liquan/data/K562/H3K4me3.bigWig')
    k4me3_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()
    
    bw=pyBigWig.open('/cluster2/huanglab/liquan/data/K562/CTCF.bigWig')
    ctcf_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()

    bw=pyBigWig.open('/cluster2/huanglab/liquan/data/K562/H3K27me3.bigWig')
    k27me3_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()
    
    fig=plt.figure(figsize=(5,3))
    
    positions = list(range(startstart,endend))

    plt.plot(positions, dnase_array/dnase_array.max()+2.0, color='#AFBA0F',alpha=0.35)
    plt.plot(positions, atac_array/atac_array.max()+1.5, color='#a29bfe',alpha=0.35)
    plt.plot(positions, k27ac_array/k27ac_array.max()+1.0, color='#70a1ff',alpha=0.35)
    plt.plot(positions, k4me3_array/k4me3_array.max()+0.5, color='#7bed9f',alpha=0.35)
    plt.plot(positions, ctcf_array/ctcf_array.max()+0.0, color='#778ca3',alpha=0.35)
    plt.plot(positions, k27me3_array/k27me3_array.max()-0.5, color='#778ca3',alpha=0.35)
    
    plt.xlim([startstart,endend])
    plt.xticks([startstart,int( (startstart+endend)/2 ),endend])
    plt.ylim([-1.0,3.5])
    
    plt.title('Genome Context')

    plt.xlabel('Genome loci')

    plt.show()


def draw_loci_context_ax(loci,feature=[],figsize=(5,3)):
    chrchr = loci[0]
    startstart = loci[1]
    endend = loci[2]

    #缩写改全称的词汇表
    epi_key_dict={'dnase':'DNase','atac':'ATAC',
                  'k27ac':'H3K27ac','k4me3':'H3K4me3',
                  'k27me3':'H3K27me3','k9me3':'H3K9me3',
                  'ctcf':'CTCF'}
    y_bottom=0
    # 创建一个 Figure 和 Axes 对象
    fig, ax = plt.subplots(figsize=figsize)
    positions = list(range(startstart, endend))
    
    # 打开 BigWig 文件并读取数据
    for e in feature:
        
        bw = pyBigWig.open(f'/cluster2/huanglab/liquan/data/K562/{ epi_key_dict[e] }.bigWig')
        array = np.array(bw.values(chrchr, startstart, endend))
        bw.close()
        
        ax.fill_between(positions, y_bottom, array/array.max()+y_bottom, color='#95afc0',alpha=0.35)
        y_bottom+=0.5

    # 设置坐标轴范围和标签
    ax.set_xlim([startstart, endend])
    ax.set_xticks([startstart, int((startstart + endend) / 2), endend])
    
    ax.set_ylim([-0.05, y_bottom+1])
    ax.set_yticks([])

    ax.set_title('Genome Context')
    ax.set_xlabel('Genome loci')

    # 返回 Axes 对象
    return ax