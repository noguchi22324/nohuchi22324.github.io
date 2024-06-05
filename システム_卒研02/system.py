import pandas as pd
import numpy as np
import itertools
import copy
from decimal import Decimal, ROUND_HALF_UP
import eel
import os
from tqdm import tqdm
import math

###############################################################################################
''' ラフ集合分析 '''
###############################################################################################

#データの属性と属性値の組み合わせを取得する。
#呼び出される順序: rough_sets(data)関数内で呼び出される。
def unique(data):                           #属性と属性値
    lis_att = []
    for i in range(len(data.columns)-1):
        lis_att.append([data.columns[i]] + sorted(list(set(list(data.iloc[:, i])))))
    return lis_att

#データフレームの列名を変更する。
#呼び出される順序: rough_sets(data)関数内で呼び出される。
def change_name(data):                      #名前変更
    for i in data.columns.values:
        data[i] = i + ":" + data[i]
    return data

#呼び出される順序: rough_sets(data)関数内で呼び出される。
def sita_comp(sita, len_sita, comp):        #下近似を求める
    drop_list = []
    drop_item = pd.merge(sita.iloc[:,:-1], comp.iloc[:,:-1])
    for i in range(len(drop_item)):
        for j in range(len_sita):
            if list(drop_item.iloc[i,:]) == list(sita.iloc[j,:-1]):
                drop_list.append(j)
    return sita.drop(sita.index[drop_list])

#呼び出される順序: rough_sets(data)関数内で呼び出される。
def make_dcmatrix(sita, comp):        #決定行列の作成
    matrix = pd.DataFrame(index=sita.index, columns=comp.index)
    for i in range(len(sita)):
        for j in range(len(comp)):
            matrix.iat[i,j] = sorted(set(sita.iloc[i][~sita.iloc[i].isin(comp.iloc[j])].tolist()))
    return matrix

#呼び出される順序: rough_sets(data)関数内で呼び出される。
def reduce_dcmatrix(matrix):                    #部分集合の大きい方を削除
    matrix = matrix.values.tolist()
    for i in range(len(matrix)):
        drop_list = []
        lis = []
        matrix[i] = sorted(matrix[i], key=len)  #長さで並び替え
        mx = copy.deepcopy(matrix[i])
        for j in itertools.combinations(range(len(mx)), 2):
            if set(mx[j[0]]) <= set(mx[j[1]]):
                drop_list.append(j[1])
        for j in range(len(mx)):
            if j not in list(set(drop_list)):
                lis.append(mx[j])
        matrix[i] = lis
    return matrix

#決定行列を展開して決定ルールを生成する。
#呼び出される順序: rough_sets(data)関数内で呼び出される。
def deploy_dcmatrix(matrix):                    #展開処理
    matrix_num = len(matrix)
    def judge_and(array, lis):                  #and結合
        for i in array:
            copy_lis = copy.deepcopy(lis)
            flag = 0
            for j in lis:
                if set(i) <= set(j):
                    copy_lis.remove(j)
                elif set(i) > set(j):
                    flag += 1
            if flag == 0:
                copy_lis.append(list(set(i)))
            lis = copy_lis
        return lis
    def judge_or(array, lis):                    #or結合
        for i in lis:
            copy_array = copy.deepcopy(array)
            flag = 0
            for j in array:
                if set(j) < set(i):
                    flag += 1
                elif set(j) >= set(i):
                    copy_array.remove(j)
            if flag == 0:
                copy_array.append(list(set(i)))
            array = copy_array
        return array
    def separate(x):                            #リストを要素ごとに分解する関数
        list = [[]] * len(x)
        for i in range(len(x)):
            list[i] = [x[i]]
        return list
    dc_list = []
    for i in range(matrix_num):
        list_01 = []
        num = len(matrix[i])
        if num > 1:
            former = separate(list(matrix[i][0]))
            for j in range(1, num):
                list_02 = []
                fl_list = []
                latter = separate(list(matrix[i][j]))
                for k in former:
                    for l in latter:
                        fl_list.append(k + l)
                former = judge_and(fl_list, list_02)
            list_01 = former
        elif num == 1:
            list_01.append(list(matrix[i][0]))
        dc_list = judge_or(dc_list, list_01)
    return dc_list

#呼び出される順序: rough_sets(data)関数内で呼び出される
def add_ci(sita, dc_rule, num, ci_len_sita, pf_uni_name):                #選好とC.I.値を求めて決定ルールリストの後ろに追加
    pf_name = pf_uni_name[num]
    dc_rule_list = []
    for i in range(len(dc_rule)):
        count = 0
        fit_list = []
        for j in range(len(sita)):
            if set(dc_rule[i]) <= set(sita.values[j].tolist()):
                count += 1
                fit_list.append(sita.index[j])
        dc_rule_list.append([dc_rule[i], pf_name, str(Decimal(str(count/ci_len_sita)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)), fit_list])
    sorted_rule = sorted(dc_rule_list, key=lambda x: x[-2], reverse=True) #C.I.値順にソート

    print(pf_name,"：",len(sorted_rule))

    if len(sorted_rule) >= 100:                #決定ルールが100以上の場合減らす
        sorted_rule = sorted_rule[:math.floor(len(sorted_rule)*0.01)]
        #sorted_rule = sorted_rule[:100]

    print(pf_name,"：",len(sorted_rule))

    return sorted_rule

#呼び出される順序: prepare(data)関数内で呼び出される。
def rough_sets(data):  # ラフ集合の計算を行い、C.I.値も求める関数
    change_name(data)
    rh = data                                                     #ラフ集合用のデータ
    pf_name = rh.columns.values[-1]                               #決定属性集合の属性の名前
    pf_uni_name = list(rh[pf_name].unique())                      #決定属性集合の属性値の名前
    pf_num = len(pf_uni_name)
    rule = []
    for num in tqdm(range(pf_num)):
        sita_01 = rh.groupby(pf_name).get_group(pf_uni_name[num]) #下近似_01
        len_sita_01 = len(sita_01)                                #下近似オリジナルの長さ
        comp_01 = rh[rh[pf_name] != pf_uni_name[num]]             #比較対象
        sita_02 = sita_comp(sita_01, len_sita_01, comp_01)        #下近似_02
        #決定行列削除
        comp_02 = comp_01.drop(comp_01.columns[-1], axis=1)
        sita_03 = sita_02.drop(sita_02.columns[-1], axis=1)
        dcmatrix_01 = make_dcmatrix(sita_03, comp_02)                            #１．決定行列作成
        dcmatrix_02 = reduce_dcmatrix(dcmatrix_01)                               #２．部分集合の大きい方を削除
        dc_rule_01 = deploy_dcmatrix(dcmatrix_02)                                #３．展開処理
        dc_rule_02 = add_ci(sita_03, dc_rule_01, num, len_sita_01, pf_uni_name)  #４．C.I.値の付与
        rule += dc_rule_02                                                       #５．決定ルールの合体
    return sorted(rule, key=lambda x:x[-2], reverse=True), pf_uni_name

#呼び出される順序: prepare(data)関数内で呼び出される。
def frequency(rule):                                              #出現頻度の算出
    result = []
    for i in tqdm(range(len(rule))):
        dc = rule[i][0]
        lis = []
        for j in range(1, len(dc) + 1):
        	for k in itertools.combinations(dc, j):
        	    lis.append(list(k))
        for j in range(len(lis)):
            flag = 0
            for k in range(len(result)):
                if set(lis[j]) == set(result[k][0]):
                    flag += 1
                    result[k][1].append(i)
            if flag == 0:
                result.append([sorted(lis[j]),[i]])
    result_02 = sorted(result, key=lambda x: len(x[-1]), reverse=True)

    slice_num = 0                                                #含まれるルールの数が１のものは除外
    for i in range(len(result_02)):
        if len(result_02[i][1]) == 1:
            slice_num = i
            break
    return result_02[:slice_num]

#呼び出される順序: py_to_js()関数内で呼び出される。
def prepare(data):                                              #jsに渡す準備
    dc_rule, pf_uni_name = rough_sets(data)
    core = frequency(dc_rule)
    core_rule = []
    for i in range(len(core)):
        lis = []
        for j in range(len(core[i][1])):
            no_core = list(set(core[i][0]) | set(dc_rule[core[i][1][j]][0]))#変更した（ ^ -> | ）
            lis.append([sorted(no_core)] + dc_rule[core[i][1][j]][1:])
        core_rule.append(sorted(lis, key=lambda x: x[1]))  #C.I.値順にソート
        
        print(i,core[i])#コアの順番とラベル表示

    return core, core_rule, dc_rule, pf_uni_name

#split_tf_idf(list_tf_idf)関数内で呼び出される。
#def tf_idf(no_core):
#    no_core_02 = copy.deepcopy(no_core)
#    list_tf_idf = []
#    for i in range(len(no_core)):
#        list_tfidf_in = []
#        for j in range(len(no_core[i])):
#            for k in range(len(no_core[i][j][0])):
#                tf = 1 / len(no_core[i][j][0])
#                count_idf = 0
#                for l in range(len(no_core[i])):
#                    if no_core[i][j][0][k] in no_core[i][l][0]:
#                        count_idf += 1
#                idf = math.log10(len(no_core[i]) / count_idf)
#                tf_idf = float(Decimal(tf*idf).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
#                no_core_02[i][j][0][k] = [no_core_02[i][j][0][k], tf_idf]
#                list_tfidf_in.append(tf_idf)
#            no_core_02[i][j][0] = sorted(no_core_02[i][j][0], key=lambda x: x[-1], reverse=True)
#        list_tf_idf.append(list_tfidf_in)
#
#    return no_core_02, list_tf_idf

#TF-IDFの計算結果を分割し、JavaScript側で使用できる形式に変換する。
#呼び出される順序: add()関数内で呼び出される。
#def split_tf_idf(list_tf_idf):
#    list_4parts = []
#    for i in range(len(list_tf_idf)):
#        list_in = np.percentile(list_tf_idf[i], q=[75, 50, 25])
#        list_4parts.append(','.join(map(str, list_in)))

#    return list_4parts

############################################################コアごとに出現する属性値のリストを得る
def get_core_attribute(core_rule):
  core_attribute = []

  for i in range(len(core_rule)):
    # 空のセットを作成
    my_set = set()
    # 決定ルールで出現した属性のリストを作成
    for j in range(len(core_rule[i])):
        # 決定ルールの属性部分のみをループ
        for k in range(len(core_rule[i][j][0])):
            # 以前に出ていない属性をセットに追加
            my_set.add(core_rule[i][j][0][k])

    # セットをリストに変換して追加
    core_attribute.append(list(my_set))

  return core_attribute

############################################################コアごとの決定ルールをサンプルとしたリストを得る,決定クラスを得る
def get_core_object(core_rule):
  core_object = []
  core_class = []#決定クラス

  #ラフ集合で算出した決定ルールを オブジェクト（サンプル）として、リストを作成
  for k in range(len(core_rule)):
    objects = []
    classes = []#決定クラス
    for i in range(len(core_rule[k])):
      label = f"コア{k+1}：決定ルール{i+1}"
      objects.append(label)
      classes.append(core_rule[k][i][1])#決定クラス

    core_object.append(objects)
    core_class.append(classes)#決定クラス

  return core_class#決定クラス

############################################################コアごとのコンテクスト表を得る
def get_core_context(core_rule):
  core_context = []

  for k in range(len(core_rule)):

    # 二次元配列を初期化
    two_dimensional_array = []
    core_attribute = get_core_attribute(core_rule)

    # 二次元配列に値を追加
    for i in range(len(core_rule[k])):  # サンプルの数だけ行を追加
        row = [False] * len(core_attribute[k])  # 新しい行を作成し、すべての要素をFalseで初期化
        for j in range(len(core_attribute[k])):  # 属性の数だけ行を追加
            for element in core_rule[k][i][0]:
                if element in core_attribute[k][j]:
                    row[j] = True
        two_dimensional_array.append(row)  # 二次元配列に行を追加

    core_context.append(two_dimensional_array)

  return core_context

###############################################################################################
''' 形式概念分析 '''
###############################################################################################

class formal_concept_analysis:

    def __init__(self, a, b, c):
        self.obj = a
        self.prop = b
        self.cont = c

        self.n_obj = self.obj
        self.n_prop = self.prop
        self.n_cont = self.cont

        self.m_extention = []  # m|n表に対応するサンプルの階層
        self.m_prop = []

        self.stratum = []
        self.included_nodes = []
        self.includes_nodes = []
        self.connection = []
        self.prop_index = []
        self.obj_index = []

    ################################ｍ｜ｍ’表を文字列で出力
    def ExportConcept(self):
        for i in range(len(self.m_prop)):
            tmp_prop = []
            for j in range(len(self.m_prop[i])):
                tmp_prop.append(self.n_prop[self.m_prop[i][j]])  # インデックス修正とappendに変更
            #print("%2d" % i, end="")  # 文字列フォーマット修正
            #print("| ", end=" ")  # |
            #print(tmp_prop, end=" ")  # 文字列フォーマット修正
            #print("|", end="")  # |

            tmp_extention = []
            for j in range(len(self.m_extention[i])):
                tmp_extention.append(self.n_obj[self.m_extention[i][j]])  # インデックス修正とappendに変更
            #print(tmp_extention)  # m'(  )

    ########################ハッセ図のノードの繋がりとラベルを編集#######################
    def prepare_d3(self):
        data1 = 0
        data2 = [[1, 4, 5, 6, 7], [3, 5, 6, 7, 2]]

        data3 = []

        for i in range(len(data1)):
            row = data1[i] + [max(data1[i]) + x for x in data2[i]]
            data3.append(row)

        print(data3)


        return 0

    ########################ハッセ図のノードの繋がりを得る#######################
    def get_edge_prop(self):
        edge_prop_per = [] #属性を繋ぐ部分のエッジ(親)
        edge_prop_chi = [] #属性を繋ぐ部分のエッジ（子）
        #connectionのリスト内の要素とそのインデックスを抽出
        #ノード間の繋がりをノード番号で記述
        for row_index, row in enumerate(self.connection):
              for col_index, element in enumerate(row):
                if element > -1:#最下層に位置する外延は排除
                   if not element == self.stratum[-1][-1]:#最下層に繋がる外延以外で考える
                        edge_prop_per.append(row_index)
                        edge_prop_chi.append(element)
        x = max(edge_prop_chi)#属性の中でラベル番号が最後のものを保存

        for index, element in enumerate(self.obj_index):#決定クラスの繋がりを加える
            edge_prop_per.append(element)
            edge_prop_chi.append(x+1+index)#属性の中で最後のラベル番号にindex（決定クラスの番号）を足す
            #indexは0からなので1を足す必要ある

        return [edge_prop_per, edge_prop_chi]
        #print(row_index, "---->",element)

   ########################ハッセ図のノードのラベル（属性）を得る#######################
    def get_node_label(self):
        node_label = []
        for i in range(len(self.m_prop)):
            tmp_prop = []
            for j in range(len(self.m_prop[i])):
                tmp_prop.append(self.n_prop[self.m_prop[i][j]])  # インデックス修正とappendに変更

            #if len(tmp_prop) >=1:#属性が１つのときのみ追加
            node_label.append(tmp_prop)
            #else :
            #    node_label.append([])

        # 含まれないインデックスに"??"を代入する
        for i in range(len(node_label)):
            if i not in self.prop_index:
                node_label[i] = []


        for i in range(len(self.n_obj)):#決定クラスのラベル追加
            node_label.append([self.n_obj[i]])

        return node_label

    ###########################コンセプト作成処理################################

    def make_concept(self):
        # m|m'リストに初期値を入力
        tmp_obj = list(range(len(self.obj)))
        self.m_extention.append(tmp_obj)

        # 空属性をm_propに入れる
        self.m_prop.append([])

        # 列ごとに外延を数えるか数えないかのフラグを格納する配列を用意
        prop_size = len(self.prop)
        del_column = [True] * prop_size

        # m|m'表を埋めていく
        while True:
            # c_propにいちばんチェックの多い属性列番号を入れる
            c_prop = self.common_prop(del_column)

            # もしc_propが-1(該当する属性なし)の場合、ループを打ち切る
            if c_prop == -1:
                break

            # コンテクスト表におけるc_prop番の属性の除外フラグをFalseにする
            del_column[c_prop] = False

            # PushExtentionに属性の番号を渡すためのリストを作成
            first_prop = [c_prop]

            # m|m'表に属性とその外延を追加
            self.push_extention(self.find_extention2(c_prop), first_prop)

        #return self.m_prop,self.n_prop

    def push_extention(self, extention, ex_prop):
        # まずはじめに、引数の外延が既出の外延リストに現れているかどうかを調べる
        for i in range(len(self.m_extention)):
            # もし既出外延とカブってたら、その既出外延の属性リストに引数の属性を加えて処理を終了
            if extention == self.m_extention[i]:
                self.m_prop[i].extend(ex_prop)
                return

        # カブってなかったら、それを新規にm|m'表の外延リストと属性リストに追加
        self.m_prop.append(ex_prop)
        self.m_extention.append(extention)

        # m|m'表に追加した外延と他の外延との共通集合を求める
        for i in range(len(self.m_extention)):
            origin = True  # 共通集合の外延と他の外延がカブってないかどうかのフラグ
            int_ext = self.intersection(extention, self.m_extention[i])  # 既出外延と引数の外延との共通集合を一時リストに入れる

            for j in range(len(self.m_extention)):
                # できた共通集合を既出外延の全てと比べる
                # もしも共通集合とすべての既出外延が異なる場合は、その共通集合に対してプッシュ処理
                if int_ext == self.m_extention[j]:
                    origin = False
                    break

            if origin:
                # 共通集合の元になった属性を合成したものを作る
                int_prop = self.m_prop[i] + ex_prop
                # 外延と属性の共通集合をm|m'表にプッシュ
                self.push_extention(int_ext, int_prop)

    def common_prop(self, del_column):
        tmp = 0  # 属性数を数えるための一時変数
        most = 0  # 最も多い属性の属性数
        most_prop = -1  # 最も多い属性の番号

        for i in range(len(self.prop)):  # 属性の数だけループ
            if del_column[i]:  # 除外フラグがTrueでない場合は処理を飛ばす
                tmp = sum(self.cont[j][i] for j in range(len(self.obj)))  # オブジェクト数の数だけループしてcontのTrue数を走査
                if tmp > most and del_column[i]:  # もしTrue数が最大かつ除外フラグがTrueだったらそれを記録
                    most = tmp
                    most_prop = i
                tmp = 0

        return most_prop

    def find_extention2(self, prop_num):
        tmp_ext = []  # 一時的な格納用のリストを作る

        for i in range(len(self.n_obj)):  # オブジェクトの数だけループ
            # propの属性のところがTrueだったら、その外延をtmp_extに加える
            if self.cont[i][prop_num]:
                tmp_ext.append(i)
        return tmp_ext

    def intersection(self,list1,list2):
        # 共通集合を格納するリストを作成
        inter = []

        # list1とlist2の要素を比較し、共通する要素をinterに追加
        for item in list1:
            if item in list2:
                inter.append(item)

        return inter



    ##############################################階層構造作成###########################################

    #外延リストm_extentionの包括関係を調べて階層リストstratumに格納する
    def make_stratum(self):
        #全外延の通し番号を格納するためのリストsを作成
        s = []
        #空集合属性の外延(インデックス0番)を除くすべての外延の番号をsにプッシュ
        for i in range(1, len(self.m_extention)):
            s.append(i)

        #空集合属性の外延(インデックス0番)をstratumにプッシュ
        zero = [0]
        self.stratum.append(zero)


        #//階層リストを作成する
        while True:
            #sの各外延とm|m'表の外延リストとの包括関係を走査する
                #m|m'表の外延リストの中にsの外延に包含されているものがあれば、それをwにプッシュ
                #こうすることで、全外延の中で、何かの外延の下位になっているものを洗い出す

                #被包括の外延番号を格納するためのリストwを作成
            w = []

            #TODO デバッグ用
                #下のループ処理が重い
                #sの全外延に対し、他の外延に対する部分集合になっていないかどうかチェック
            for i in range(len(s)):
                for j in range(len(s)):
                    #もしm_extentionにsの外延の部分集合になっているものがあれば、それをwに加える
                    if self.is_proper_subset(self.m_extention[s[j]], self.m_extention[s[i]]):
                        #m_extentionの外延番号を格納するための一時リストを作成し、外延番号を格納
                        sub_extention = [s[i]]
                        #一時変数を作り、作成した一時リストとwの和集合を格納する
                        tmp_w = self.union(sub_extention, w)
                        #できた和集合をwの中身と置き換える
                        w = tmp_w

            #wとsの差集合rをとる
            #rは他の外延のに包含されていない→階層が上の外延となる
            r = self.difference(s, w)

            #もしrが空集合の場合は処理を打ち切る
            if len(r) == 0:
                break

            #rを階層リストstratumにプッシュ
            self.stratum.append(r)
            #sをwで置き換える
            s = w

    #外延同士のつながりを計算し、connectionに格納する
    def make_connection(self):
        #ハッセ図全体のノードの包含・被包含関係、各ノードが下位のどのノードとつながっているかを格納するリストを用意
        for i in range(len(self.m_extention)):
            #ノードの数だけ空のリストを用意してリストに追加
            self.included_nodes.append([])
            self.includes_nodes.append([])
            self.connection.append([])

        #stratumの各階層の外延ごとに計算を行い、どの下位外延につながりがあるかを調べる
        #全ノードをstratumの最上層から順に走査する
        #iが階層、jがその階層内のノードの順番
        for i in range(len(self.stratum)):
            for j in range(len(self.stratum[i])):
                #現在のノードの番号
                high = self.stratum[i][j]

                #もし階層が最下層だったらどこにも繋がっていないので、-1を入れる
                if i == len(self.stratum) - 1:
                    self.connection[high].append(-1)
                else:
                    #最下層でない場合、stratumの下位階層にある外延の数だけ包括関係を調べる
                    for k in range(len(self.stratum[i + 1])):
                        #highに包含されている側の下位のノード番号
                        low = self.stratum[i + 1][k]
                        #もし包括関係にあったら、connectionリストにその番号を格納
                        if self.is_subset(self.m_extention[high], self.m_extention[low]):
                            self.connection[high].append(low)
                            #各lowノードに、このノードを包含しているノードとさらにそのノードを包含しているノード…の番号を格納
                            self.included_nodes[low] = self.union(self.included_nodes[high], self.included_nodes[low])
                            self.included_nodes[low].append(high)

        #階層が２つ以上離れているノードどうしの包含関係を調べる
        #2階層上のノードと比較→3階層上のノードと比較……と進んでいく
        for up_count in range(2, len(self.stratum)):
            #全ノードをstratumの最下層から順に走査する
            #iが階層、jがその階層内のノードの順番
            for i in range(len(self.stratum) - 1, -1, -1):
                for j in range(len(self.stratum[i])):
                    #今いるノードからup_count上の階層を走査する
                    #もし今いるノードのincluded_nodesに入ってない＝包含関係を検証していないノードがあれば包含関係を検証
                    #i=階層がup_countより小さい場合はそれ以上上には進めないので処理をパスする
                    if i > up_count:
                        #現在のノード番号
                        current_node = self.stratum[i][j]
                        #included_nodesと階層のノード情報の差集合をとって未検証のノードだけを残し、それを一時リストに格納
                        unverified_node = self.difference(self.stratum[i - up_count], self.included_nodes[current_node])
                        #一時リストに残った未検証ノードひとつひとつについて包含関係を検証
                        for k in range(len(unverified_node)):
                            #検証対象の上位ノード番号
                            high_node = unverified_node[k]
                            if self.is_subset(self.m_extention[high_node], self.m_extention[current_node]):
                                #包含関係にあった場合、connectionとincludedに情報を追加
                                self.connection[high_node].append(current_node)
                                #included_nodesの各ノードに、このノードを包含しているノードとさらにそのノードを包含しているノード…の番号を格納
                                self.included_nodes[current_node] = self.union(self.included_nodes[high_node], self.included_nodes[current_node])
                                self.included_nodes[current_node].append(high_node)

        #TODO 急造なのであとで修正する
        #もう一回全ノードをstratumの最下層から順に走査して下位ノードの一覧リストを作る
        #iが階層、jがその階層内のノードの順番
        for i in range(len(self.stratum) - 1, -1, -1):
            for j in range(len(self.stratum[i])):
                low = self.stratum[i][j]
                if i > 0:
                    for k in range(len(self.connection)):
                        if low in self.connection[k]:
                            self.includes_nodes[k] = self.union(self.includes_nodes[low], self.includes_nodes[k])
                            self.includes_nodes[k].append(low)

    #属性と外延に対応するノードを見つける
    def make_index(self):

        #計算に使う変数とか
        global prop_index  # prop_indexをグローバル変数として宣言
        global obj_index  # obj_indexをグローバル変数として宣言

        prop_size = len(self.n_prop)     #属性数
        obj_size = len(self.n_obj)       #オブジェクト数
        stratum_size = len(self.stratum) #Hasse図の階層数

        #計算結果を格納する配列のアドレスを確保し、配列を-1で初期化
        self.prop_index = [-1] * prop_size
        self.obj_index = [-1] * obj_size


        defined = 0  #発見した属性やオブジェクトの数を格納するカウンタ

        #属性を調べる
        #ノードを上の階層から順に走査し、属性が最初に現れるノードの番号を調べる。
        #全属性の出現ポイントが判明したら計算を終了
        for i in range(stratum_size):
            for j in range(len(self.stratum[i])):
                #ノードの番号
                node_num = self.stratum[i][j]

                #ノードの外延一つ一つを調べる。
                #未発見の属性があれば配列に追加。
                for k in range(len(self.m_prop[node_num])):
                    #属性の番号
                    prop_num = self.m_prop[node_num][k]

                    #prop_indexにおける該当の属性が-1 =未発見の属性だった場合はその属性を配列に追加
                    if self.prop_index[prop_num] == -1:
                        self.prop_index[prop_num] = node_num
                        #カウンタを加算
                        defined += 1

                    if defined >= prop_size:
                        break

                if defined >= prop_size:
                    break

            if defined >= prop_size:
                break

        #オブジェクトを調べる
        #カウンタをリセット
        defined = 0

        #ノードを下の階層から順に走査し、オブジェクトが最初に現れるノードの番号を調べる。
        #全オブジェクトの出現ポイントが判明したら計算を終了
        #属性と違って今度は下から調べる
        for i in range(stratum_size - 1, -1, -1):
            for j in range(len(self.stratum[i])):
                #ノードの番号
                node_num = self.stratum[i][j]

                #ノードの内包一つ一つを調べる。
                #未発見の属性があれば配列に追加。
                for k in range(len(self.m_extention[node_num])):
                    #オブジェクトの番号
                    prop_num = self.m_extention[node_num][k]

                    #obj_indexにおける該当の属性が-1 =未発見の属性だった場合はその属性を配列に追加
                    if self.obj_index[prop_num] == -1:
                        self.obj_index[prop_num] = node_num
                        #カウンタを加算
                        defined += 1

                    if defined >= obj_size:
                        break

                if defined >= obj_size:
                    break

            if defined >= obj_size:
                break


    #########################################形式概念分析で計算する用の関数###################

    def union(self,list1, list2):
        # 和集合を格納するリストを作成
        uni = list(list1)

        # list2の要素のうち、list1にない要素をuniに追加
        for item in list2:
            if item not in uni:
                uni.append(item)

        return uni


    def difference(self,list1, list2):
        # 差集合を格納するリストを作成
        diff = list(list1)

        # list2の要素を取り除く
        for item in list2:
            if item in diff:
                diff.remove(item)

        return diff


    def intersection(self,list1, list2):
        # 共通集合を格納するリストを作成
        inter = []

        # list1とlist2の要素を比較し、共通する要素をinterに追加
        for item in list1:
            if item in list2:
                inter.append(item)

        return inter


    def is_subset(self,super_set, sub_set):
        # sub_setがsuper_setの部分集合であればTrueを返す
        return all(item in super_set for item in sub_set)


    def is_proper_subset(self,super_set, sub_set):
        # sub_setがsuper_setの真部分集合であればTrueを返す
        return all(item in super_set for item in sub_set) and len(super_set) > len(sub_set)


    def is_eq(self,super_set, sub_set):
        # sub_setがsuper_setと要素が等しい場合にTrueを返す
        return sorted(super_set) == sorted(sub_set)


    def is_proper_eq(self,super_set, sub_set):
        # sub_setがsuper_setと要素の順番まで含めて等しい場合にTrueを返す
        return super_set == sub_set


    def search(self,list, s):
        # 2次元リストから要素sの含まれている一次インデックスを検索して返す
        for i, sublist in enumerate(list):
            if s in sublist:
                return i
        return -1


    def replace(self,list, a, b):
        # 2次元リストのインデックス番号aとbの要素を入れ替える
        list[a], list[b] = list[b], list[a]


###############################################################################################
''' JS接続 '''
###############################################################################################
@eel.expose
def add(arr, ind, col):#引数
    data = pd.DataFrame(arr, index=ind, columns=col)
    core, core_rule, dc_rule, pf_uni_name = prepare(data)

    #value = prepare(data) #ラフ集合を行って決定ルールを得る

    #core = value[0] #コア
    #core_rule = value[1] #コアで分類して括った決定ルール
    #dc_rule = value[2] #コアで分ける前の決定ルール
    #pf_uni_name = value[3] #決定クラス　ex['選好:好き', '選好:どちらでもない']

    core_attribute = get_core_attribute(core_rule) #属性
    core_object = get_core_object(core_rule) #オブジェクト、決定クラス
    core_context = get_core_context(core_rule) #コンテクスト表

    core_node_label = []#コアで括ったノードのラベル（おそらく一番最初がコア）
    core_edge_prop = []#コアで括った決定ルールの条件部のみを階層化したときのノード番号のペア[上位ノード,下位ノード]の順

    for i in range(len(core_object)):
        a = core_object[i]
        b = core_attribute[i]
        c = core_context[i]

        # クラスのインスタンスを作成
        concept_lattice = formal_concept_analysis(a, b, c)
        #この順番に実行しないとデータ受け渡しが上手くいかない
        #形式概念分析を行う
        concept_lattice.make_concept()
        concept_lattice.make_stratum()
        concept_lattice.make_connection()
        concept_lattice.make_index()
        concept_lattice.ExportConcept()
        #ハッセ図に必要なノードの繋がりとラベルを得る
        node_label = concept_lattice.get_node_label()
        edge_prop = concept_lattice.get_edge_prop()

        core_node_label.append(node_label)
        core_edge_prop.append(edge_prop)

    #no_core, list_tf_idf = tf_idf(no_core)
    #list_4parts = split_tf_idf(list_tf_idf)
    print(core_node_label[0])
    print(core_edge_prop[0])

    data1 = core_edge_prop[0]
    data2 = core_edge_prop[1]

    data3 = []

    for i in range(len(data1)):
        row = data1[i] + [max(data1[1])+1 + x for x in data2[i]]
        data3.append(row)
    
    print(data3)


    eel.js_func(core_node_label[:300], core_edge_prop[:300],pf_uni_name,data3,core)#戻り値

    #eel.js_func(core[:300], no_core[:300], pf_uni_name)#戻り値

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'Output')


#PythonのデータをJavaScriptに渡すための形式に変換する。
#各関数を順番に呼び出し、結果をJavaScript側で使用できる形式に変換する
def py_to_js():
    eel.init(path)
    eel.start("index.html", size=(1024, 768), port=8080)

if __name__ == "__main__":
    py_to_js()
