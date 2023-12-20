import csv
import pandas as pd
import numpy as np
import re
import statistics
import math
import heapq
from itertools import product
from datetime import datetime

Price_of_Underlying1 = 436.93
TradeSizeFactor = 100
ir = []

result = []
sublist = []
results = []
results1_ = []
results2_ = []
prices_c = []
prices_p = []

all_top_10_c = []
all_top_10_p = []
all_top_10 = []

Max_DDs_c = []
Max_DDs_p = []

MaxDaysinDD_c = []
MaxDaysinDD_p = []
MaxDaysinDD = []

MeanDaysInRecovery_cs = []
MeanDaysInRecovery_ps = []
MeanDaysInRecovery_s = []

MaxDaysInLoosingStreak_cs = []
MaxDaysInLoosingStreak_ps = []
MaxDaysInLoosingStreak_s = []

MeanDaysInLoosingStreakLocalMax_cs = []
MeanDaysInLoosingStreakLocalMax_ps = []
MeanDaysInLoosingStreakLocalMax_s = []

PnLs_calls = []
PnLs_puts = []

Equity_0 = 10000

Rtns_c = []
Rtns_p = []

RtnAnns_c = []
RtnAnns_p = []


Rtns = []
RtnAnns = []
Max_DDs = []
Calm_simples = []

Calm_simple_cs = []
Calm_simple_ps = []

mins = []
maxs = []
means = []
l_means = []
p_means = []
p_mean_l_means = []
mean_mins = []
median_mins = []
medians = []
p_counts = []
n_counts = []
p_sums = []
n_sums = []
prob_of_profits = []
oddss = []

mins_c = []
maxs_c = []
means_c = []
l_means_c = []
p_means_c = []
p_mean_l_means_c = []
mean_mins_c = []
median_mins_c = []
medians_c = []
p_counts_c = []
n_counts_c = []
p_sums_c = []
n_sums_c = []
prob_of_profits_c = []
oddss_c = []

mins_p = []
maxs_p = []
means_p = []
l_means_p = []
p_means_p = []
p_mean_l_means_p = []
mean_mins_p = []
median_mins_p = []
medians_p = []
p_counts_p = []
n_counts_p = []
p_sums_p = []
n_sums_p = []
prob_of_profits_p = []
oddss_p = []


def DaysIRecoveryForDates(mylist, dates):
    DaysInRecoveryForDates = [float("NaN")]
    for indx, xx in enumerate(mylist):
        if indx + 1 < len(mylist):
            if mylist[indx+1] == 0:
                if mylist[indx] == 0:
                    DaysInRecoveryForDates.append(float("NaN"))
                else:
                    DaysInRecoveryForDates.append(mylist[indx])
            else:
                DaysInRecoveryForDates.append(float("NaN"))
        if indx == len(mylist) - 1 and mylist[indx] != 0:
            DaysInRecoveryForDates.append(mylist[indx])
    resulting = dict(zip(dates, DaysInRecoveryForDates))
    return resulting


def top_n_largest_without_nan(dictionary, n):
    heap = [(-float('inf'), None)] * n
    heapq.heapify(heap)

    for date, num in dictionary.items():
        if not math.isnan(num) and num > heap[0][0]:
            heapq.heappop(heap)
            heapq.heappush(heap, (num, date))

    return sorted(heap, reverse=True, key=lambda x: x[0])


def DaysIRecovery(mylist):
    DaysInRecovery = []
    for indx, x in enumerate(mylist):
        if indx + 1 < len(mylist):
            if mylist[indx + 1] == 0:
                if mylist[indx] == 0:
                    continue
                else:
                    DaysInRecovery.append(mylist[indx])
            else:
                continue
        if indx == len(mylist) - 1 and mylist[indx] != 0:
            DaysInRecovery.append(mylist[indx])
    return DaysInRecovery


def DaysILoosingStreak(mylist):
    DaysInLoosingStreak = []
    ink = 0
    for indx, x in enumerate(mylist):
        if x < 0:
            ink += 1
        else:
            ink = 0
        DaysInLoosingStreak.append(ink)
    return DaysInLoosingStreak


def DaysILoosingStreakLocalMax(mylist):
    DaysInLoosingStreakLocalMax = []
    for indx, x in enumerate(mylist):
        if indx + 1 < len(mylist):
            if mylist[indx + 1] == 0:
                if mylist[indx] == 0:
                    continue
                else:
                    DaysInLoosingStreakLocalMax.append(mylist[indx])
            else:
                continue
        if indx == len(mylist) - 1 and mylist[indx] != 0:
            DaysInLoosingStreakLocalMax.append(mylist[indx])
    return DaysInLoosingStreakLocalMax



def count_neg_pos(arr):
    pos_count, neg_count = 0, 0
    sum_of_neg, sum_of_pos = 0, 0
    for num in arr:
        if num >= 0:
            pos_count += 1
            sum_of_pos += num
        else:
            neg_count +=1
            sum_of_neg += num
    return pos_count, neg_count, sum_of_pos, sum_of_neg

month_dict = {
    'JAN': '01',
    'FEB': '02',
    'MAR': '03',
    'APR': '04',
    'MAY': '05',
    'JUN': '06',
    'JUL': '07',
    'AUG': '08',
    'SEP': '09',
    'OCT': '10',
    'NOV': '11',
    'DEC': '12'
}

df_prices = pd.read_csv('SPY.csv')
df_prices['ChFrac'] = df_prices['Adj Close'].diff() / df_prices['Adj Close'].shift(1)

with open("2023-11-08-StockAndOptionQuoteForSPY.csv") as fd:
    reader = csv.reader(fd)
    ir.append([row for idx, row in enumerate(reader) if idx > 13])

for item in ir[0]:
    if not item:
        if sublist:
            result.append(sublist)
            sublist = []
    else:
        sublist.append(item)
if sublist:
    result.append(sublist)

for l in result:
    l.pop(0)

for i in range(1, len(result)):
    df = pd.DataFrame(result[i])
    new_columns = ['','','Theo Price', 'Mark', 'BID_CALL', 'BX', 'ASK_CALL', 'AX', 'Exp', 'Strike', 'BID_PUT', 'BX', 'ASK_PUT', 'AX', 'Theo Price', 'Mark', '', '']
    df.columns = new_columns
    df = df[1:]

    for index, row in df.iterrows():
        exp = row['Exp']
        strike = row['Strike']
        bid1 = row['BID_CALL']
        ask1 = row['ASK_CALL']
        bid2 = row['BID_PUT']
        ask2 = row['ASK_PUT']

        type1 = 'C'
        type2 = 'P'

        day, month_abbr, year_str = exp.split()
        if len(day) == 1:
            day = '0' + day

        year = "20" + year_str

        month = month_dict[month_abbr]
        new_date_str = f"{year}-{month}-{day}"

        data1 = {
            'Snapshot Date': '2023-11-08',
            'Symbol': 'SPY',
            'DoE': new_date_str,
            'Type': type1,
            'Strike': strike,
            'Bid': bid1,
            'Ask': ask1
        }

        data2 = {
            'Snapshot Date': '2023-11-08',
            'Symbol': 'SPY',
            'DoE': new_date_str,
            'Type': type2,
            'Strike': strike,
            'Bid': bid2,
            'Ask': ask2
        }
        results.append(data1)
        results.append(data2)

final_df = pd.DataFrame(results)

final_df = final_df.drop(np.where(final_df['DoE'] != '2023-11-08')[0])

final_df.insert(loc=1, column='ID', value=final_df.apply(lambda row: ''.join(map(str, row[3:5])), axis=1))

final_df = final_df.drop(['Snapshot Date','Symbol', 'DoE', 'Type', 'Strike'], axis=1)

pattern = re.compile(r'(C|P)(\d+)')

print(datetime.now())

PnLss_call = dict()
PnLss_call1 = dict()
PnLss_put = dict()
PnLss_put1 = dict()
my_cb_dict = dict()
my_pb_dict = dict()
my_cs_dict = dict()
my_ps_dict = dict()
Opt1 = []
Opt2 = []
AA = []
BB = []
CC = []
DD = []
PNL_AT_ZERO_CB = []
PNL_AT_ZERO_PB = []
PNL_AT_ZERO_CS = []
PNL_AT_ZERO_PS = []
PNL_CS = dict()
PNL_CB = dict()
PNL_PS = dict()
PNL_PB = dict()
for index, row in final_df.iterrows():
    c = 0
    p = 0
    i_c = 0
    i_p = 0
    match = pattern.search(row['ID'])
    PnLs = []
    DD_c = [0]
    DD_p = [0]
    DaysinDD_c = []
    DaysinDD_p = []
    PnLs_call = []
    PnLs_call1 = []
    PnLs_put = []
    PnLs_put1 = []
    Equity_c = [Equity_0]
    Equity_p = [Equity_0]
    Price2_c = 0
    Price2_p = 0
    type_ = match.group(1)
    strike = int(match.group(2))
    Price1_call = 0
    Price1_put = 0
    Price11_call = 0
    Price11_put = 0
    if type_ == 'C':
        Price1_call = float(row['Ask'])
        my_cb_dict.update({row['ID']: Price1_call})
        Price11_call = float(row['Bid'])
        my_cs_dict.update({row['ID']: Price11_call})
        prices_c.append(Price1_call)
        PNL_AT_ZERO_CB.append(Price_of_Underlying1 - strike - Price1_call)
        PNL_CB.update({row['ID']: PNL_AT_ZERO_CB[-1]})
        PNL_AT_ZERO_CS.append(Price_of_Underlying1 - strike - Price11_call)
        PNL_CS.update({row['ID']: PNL_AT_ZERO_CS[-1]})
    if type_ == 'P':
        Price1_put = float(row['Ask'])
        my_pb_dict.update({row['ID']: Price1_put})
        Price11_put = float(row['Bid'])
        my_ps_dict.update({row['ID']: Price11_put})
        prices_p.append(Price1_put)
        PNL_AT_ZERO_PB.append(strike - Price_of_Underlying1 - Price1_put)
        PNL_PB.update({row['ID']: PNL_AT_ZERO_PB[-1]})
        PNL_AT_ZERO_PS.append(strike - Price_of_Underlying1 - Price11_put)
        PNL_PS.update({row['ID']: PNL_AT_ZERO_PS[-1]})
    for index1, row1 in df_prices.iterrows():
        if math.isnan(row1['ChFrac']):
            continue
        if type_ == 'C':
            Price2_c = ((Price_of_Underlying1 * (1 + row1['ChFrac'])) - strike)
            if Price2_c < 0:
                Price2_c = 0
            PnLs_call.append(Price2_c - Price1_call)
            PnLs_call1.append(Price2_c - Price11_call)
            equity_i_c = Equity_c[-1]+(PnLs_call[-1]*((Equity_c[-1]/TradeSizeFactor)/Price1_call))
            Equity_c.append(equity_i_c)
            DD_c.append(Equity_c[-1]/max(Equity_c) - 1)
            if DD_c[-1] != 0:
                i_c += 1
            elif DD_c[-1] == 0:
                i_c = 0
            DaysinDD_c.append(i_c)
        if type_ == 'P':
            Price2_p = (strike - (Price_of_Underlying1 * (1 + row1['ChFrac'])))
            if Price2_p < 0:
                Price2_p = 0
            PnLs_put.append(Price2_p - Price1_put)
            PnLs_put1.append(Price2_p - Price11_put)
            equity_i_p = Equity_p[-1]+(PnLs_put[-1]*((Equity_p[-1]/TradeSizeFactor)/Price1_put))
            Equity_p.append(equity_i_p)
            DD_p.append(Equity_p[-1]/max(Equity_p) - 1)
            if DD_p[-1] != 0:
                i_p += 1
            elif DD_p[-1] == 0:
                i_p = 0
            DaysinDD_p.append(i_p)

    if type_ == 'C':
        PnLss_call.update({row['ID']: PnLs_call})
        PnLss_call1.update({row['ID']: PnLs_call1})
        DaysInRecoveryForDates_c = DaysIRecoveryForDates(DaysinDD_c, df_prices["Date"])
        top_10_c = top_n_largest_without_nan(DaysInRecoveryForDates_c, 10)
        all_top_10_c.append(top_10_c)
        DaysInRecovery_c = DaysIRecovery(DaysinDD_c)
        MeanDaysInRecovery_c = statistics.mean(DaysInRecovery_c)
        DaysInLoosingStreak_c = DaysILoosingStreak(PnLs_call)
        DaysInLoosingStreakLocalMax_c = DaysILoosingStreakLocalMax(DaysInLoosingStreak_c)
        MaxDaysInLoosingStreak_c = max(DaysInLoosingStreak_c)
        MeanDaysInLoosingStreakLocalMax_c = statistics.mean(DaysInLoosingStreakLocalMax_c)
        Rtn_c = Equity_c[-1] / Equity_c[0] - 1
        RtnAnn_c = (Rtn_c + 1) ** (252/len(PnLs_call)) - 1
        MaxDDc = min(DD_c)
        if MaxDDc == 0:
            Calm_simple_c = 0
        else:
            Calm_simple_c = RtnAnn_c / MaxDDc
        Rtns_c.append(Rtn_c)
        RtnAnns_c.append(RtnAnn_c)
        Max_DDs_c.append(MaxDDc)
        Calm_simple_cs.append(Calm_simple_c)
        minimum_c = min(PnLs_call) * 100
        maximum_c = max(PnLs_call) * 100
        mean_c = statistics.mean(PnLs_call) * 100
        median_c = statistics.median(PnLs_call) * 100
        p_count_c, n_count_c, p_sum_c, n_sum_c = count_neg_pos(PnLs_call)
        prob_of_profit_c = p_count_c / len(PnLs_call)
        if n_sum_c == 0:
            n_sum_c = 1
        odds_c = p_sum_c / abs(n_sum_c)
        if n_count_c == 0:
            n_count_c = 1
        l_mean_c = (n_sum_c / n_count_c) * 100
        if p_count_c == 0:
            p_count_c = 1
        p_mean_c = (p_sum_c / p_count_c) * 100

        Opt1.append(1)
        mins_c.append(minimum_c)
        maxs_c.append(maximum_c)
        means_c.append(mean_c)
        l_means_c.append(l_mean_c)
        p_means_c.append(p_mean_c)
        p_mean_l_means_c.append(p_mean_c/l_mean_c)
        medians_c.append(median_c)
        mean_mins_c.append(mean_c / minimum_c)
        median_mins_c.append(median_c / minimum_c)
        p_counts_c.append(p_count_c)
        n_counts_c.append(n_count_c)
        p_sums_c.append(p_sum_c * 100)
        n_sums_c.append(n_sum_c * 100)
        prob_of_profits_c.append(prob_of_profit_c)
        oddss_c.append(odds_c)
        MaxDaysinDD_c.append(max(DaysinDD_c))
        MeanDaysInRecovery_cs.append(MeanDaysInRecovery_c)
        MaxDaysInLoosingStreak_cs.append(MaxDaysInLoosingStreak_c)
        MeanDaysInLoosingStreakLocalMax_cs.append(MeanDaysInLoosingStreakLocalMax_c)

        c += 1

    if type_ == 'P':
        PnLss_put.update({row['ID']: PnLs_put})
        PnLss_put1.update({row['ID']: PnLs_put1})
        DaysInRecoveryForDates_p = DaysIRecoveryForDates(DaysinDD_p, df_prices["Date"])
        top_10_p = top_n_largest_without_nan(DaysInRecoveryForDates_p, 10)
        all_top_10_p.append(top_10_p)
        DaysInRecovery_p = DaysIRecovery(DaysinDD_p)
        MeanDaysInRecovery_p = statistics.mean(DaysInRecovery_p)
        DaysInLoosingStreak_p = DaysILoosingStreak(PnLs_put)
        DaysInLoosingStreakLocalMax_p = DaysILoosingStreakLocalMax(DaysInLoosingStreak_p)
        MaxDaysInLoosingStreak_p = max(DaysInLoosingStreak_p)
        MeanDaysInLoosingStreakLocalMax_p = statistics.mean(DaysInLoosingStreakLocalMax_p)
        Rtn_p = Equity_p[-1] / Equity_p[0] - 1
        RtnAnn_p = (Rtn_p + 1) ** (252 / len(PnLs_put)) - 1
        MaxDDp = min(DD_p)
        if MaxDDp == 0:
            Calm_simple_p = 0
        else:
            Calm_simple_p = RtnAnn_p / MaxDDp
        Rtns_p.append(Rtn_p)
        RtnAnns_p.append(RtnAnn_p)
        Max_DDs_p.append(MaxDDp)
        Calm_simple_ps.append(Calm_simple_p)
        minimum_p = min(PnLs_put) * 100
        maximum_p = max(PnLs_put) * 100
        mean_p = statistics.mean(PnLs_put) * 100
        median_p = statistics.median(PnLs_put) * 100
        p_count_p, n_count_p, p_sum_p, n_sum_p = count_neg_pos(PnLs_put)
        prob_of_profit_p = p_count_p / len(PnLs_put)
        if n_sum_p == 0:
            n_sum_p = 1
        odds_p = p_sum_p / abs(n_sum_p)
        if n_count_p == 0:
            n_count_p = 1
        l_mean_p = (n_sum_p / n_count_p) * 100
        if p_count_p == 0:
            p_count_p = 1
        p_mean_p = (p_sum_p / p_count_p) * 100

        Opt2.append(1)
        mins_p.append(minimum_p)
        maxs_p.append(maximum_p)
        means_p.append(mean_p)
        l_means_p.append(l_mean_p)
        p_means_p.append(p_mean_p)
        p_mean_l_means_p.append(p_mean_p / l_mean_p)
        medians_p.append(median_p)
        mean_mins_p.append(mean_p / minimum_p)
        median_mins_p.append(median_p / minimum_p)
        p_counts_p.append(p_count_p)
        n_counts_p.append(n_count_p)
        p_sums_p.append(p_sum_p * 100)
        n_sums_p.append(n_sum_p * 100)
        prob_of_profits_p.append(prob_of_profit_p)
        oddss_p.append(odds_p)
        MaxDaysinDD_p.append(max(DaysinDD_p))
        MeanDaysInRecovery_ps.append(MeanDaysInRecovery_p)
        MaxDaysInLoosingStreak_ps.append(MaxDaysInLoosingStreak_p)
        MeanDaysInLoosingStreakLocalMax_ps.append(MeanDaysInLoosingStreakLocalMax_p)
        AA.append(float("NaN"))
        BB.append(float("NaN"))
        CC.append(float("NaN"))
        DD.append(float("NaN"))

        p += 1

    if type_ == 'C':
        data1_ = {
            'ID1': f"b{row['ID']}",
            'PnLs': PnLs_call,
        }
        results1_.append(data1_)
    if type_ == 'P':
        data2_ = {
            'ID2': f"b{row['ID']}",
            'PnLs': PnLs_put
        }
        results2_.append(data2_)

my_df1 = pd.DataFrame(results1_)
my_df2 = pd.DataFrame(results2_)
temp_df = pd.DataFrame()

sort_str = []
for ind, row in my_df1.iterrows():
    U = 437
    input_string = row['ID1']
    letter_part = input_string[0]
    number_part = int(input_string[2:])
    updated_number = number_part - U

    updated_string = f'{letter_part}{updated_number:+}'
    sort_str.append(updated_string)

my_df1['Options Count'] = Opt1
my_df1['ID2'] = sort_str
my_df1['PNLatZeroCh'] = PNL_AT_ZERO_CB
my_df1["Max"] = maxs_c
my_df1["Min"] = mins_c
my_df1["Mean"] = means_c
my_df1["Median"] = medians_c
my_df1["Mean/Min"] = mean_mins_c
my_df1["Median/Min"] = median_mins_c
my_df1["LCount"] = n_counts_c
my_df1["PCount"] = p_counts_c
my_df1["LSum"] = n_sums_c
my_df1["PSum"] = p_sums_c
my_df1["LMean"] = l_means_c
my_df1["PMean"] = p_means_c
my_df1["PMean/LMean"] = p_mean_l_means_c
my_df1["PoP"] = prob_of_profits_c
my_df1["Odds"] = oddss_c
my_df1["Rtn"] = Rtns_c
my_df1["RtnAnn"] = RtnAnns_c
my_df1["MaxDD"] = Max_DDs_c
my_df1["MaxDaysInDD"] = MaxDaysinDD_c
my_df1["MeanDaysInRecovery"] = MeanDaysInRecovery_cs
my_df1["MaxDaysInLoosingStreak"] = MaxDaysInLoosingStreak_cs
my_df1["MeanDaysInLoosingStreakLocalMax"] = MeanDaysInLoosingStreakLocalMax_cs
my_df1["Calmar Simplified"] = Calm_simple_cs
for i, top_10 in enumerate(all_top_10_c):
    numbers = [num for num, date in top_10]
    dates = [date for num, date in top_10]

    df111 = pd.DataFrame({f'DaysInRecoveryRecord{i + 1}': [numbers[i]] for i in range(10)})
    df_dates111 = pd.DataFrame({f'DaysInRecoveryDateRecord{i + 1}': [dates[i]] for i in range(10)})

    df222 = pd.concat([df111, df_dates111], axis=1)

    temp_df = pd.concat([temp_df, df222], axis=0, ignore_index=True)

my_df1 = pd.concat([my_df1, temp_df], axis=1)
my_df1["AA"] = AA
my_df1["BB"] = BB
my_df1["CC"] = CC
my_df1["DD"] = DD

my_df1["PNLatZeroCh"] = my_df1["PNLatZeroCh"].round(2)
my_df1["Max"] = my_df1["Max"].round(2)
my_df1["Min"] = my_df1["Min"].round(2)
my_df1["Mean"] = my_df1["Mean"].round(2)
my_df1["Median"] = my_df1["Median"].round(2)
my_df1["Mean/Min"] = my_df1["Mean/Min"] * -1
my_df1["Mean/Min"] = my_df1["Mean/Min"].round(2)
my_df1["Median/Min"] = my_df1["Median/Min"] * -1
my_df1["Median/Min"] = my_df1["Median/Min"].round(2)
my_df1["LCount"] = my_df1["LCount"].round(2)
my_df1["PCount"] = my_df1["PCount"].round(2)
my_df1["LSum"] = my_df1["LSum"].round(2)
my_df1["PSum"] = my_df1["PSum"].round(2)
my_df1["LMean"] = my_df1["LMean"].round(2)
my_df1["PMean"] = my_df1["PMean"].round(2)
my_df1["PMean/LMean"] = my_df1["PMean/LMean"] * -1
my_df1["PMean/LMean"] = my_df1["PMean/LMean"].round(2)
my_df1["PoP"] = my_df1["PoP"].round(2)
my_df1["Odds"] = my_df1["Odds"].round(2)
my_df1["Rtn"] = my_df1["Rtn"].round(2)
my_df1["RtnAnn"] = my_df1["RtnAnn"].round(2)
my_df1["MaxDD"] = my_df1["MaxDD"].round(2)
my_df1["MeanDaysInRecovery"] = my_df1["MeanDaysInRecovery"].round(2)
my_df1["MaxDaysInLoosingStreak"] = my_df1["MaxDaysInLoosingStreak"].round(2)
my_df1["MeanDaysInLoosingStreakLocalMax"] = my_df1["MeanDaysInLoosingStreakLocalMax"].round(2)
my_df1["Calmar Simplified"] = my_df1["Calmar Simplified"] * -1
my_df1["Calmar Simplified"] = my_df1["Calmar Simplified"].round(2)

temp_df = pd.DataFrame()
sort_str = []
for ind, row in my_df2.iterrows():
    U = 437
    input_string = row['ID2']
    letter_part = input_string[0]
    number_part = int(input_string[2:])
    updated_number = number_part - U

    updated_string = f'{letter_part}{updated_number:+}'
    sort_str.append(updated_string)

my_df2['Options counts'] = Opt2
my_df2['ID3'] = sort_str
my_df2['PNLatZeroCh'] = PNL_AT_ZERO_PB
my_df2["Max"] = maxs_p
my_df2["Min"] = mins_p
my_df2["Mean"] = means_p
my_df2["Median"] = medians_p
my_df2["Mean/Min"] = mean_mins_p
my_df2["Median/Min"] = median_mins_p
my_df2["LCount"] = n_counts_p
my_df2["PCount"] = p_counts_p
my_df2["LSum"] = n_sums_p
my_df2["PSum"] = p_sums_p
my_df2["LMean"] = l_means_p
my_df2["PMean"] = p_means_p
my_df2["PMean/LMean"] = p_mean_l_means_p
my_df2["PoP"] = prob_of_profits_p
my_df2["Odds"] = oddss_p
my_df2["Rtn"] = Rtns_p
my_df2["RtnAnn"] = RtnAnns_p
my_df2["MaxDD"] = Max_DDs_p
my_df2["MaxDaysInDD"] = MaxDaysinDD_p
my_df2["MeanDaysInRecovery"] = MeanDaysInRecovery_ps
my_df2["MaxDaysInLoosingStreak"] = MaxDaysInLoosingStreak_ps
my_df2["MeanDaysInLoosingStreakLocalMax"] = MeanDaysInLoosingStreakLocalMax_ps
my_df2["Calmar Simplified"] = Calm_simple_ps
for i, top_10 in enumerate(all_top_10_p):
    numbers = [num for num, date in top_10]
    dates = [date for num, date in top_10]

    df111 = pd.DataFrame({f'DaysInRecoveryRecord{i + 1}': [numbers[i]] for i in range(10)})
    df_dates111 = pd.DataFrame({f'DaysInRecoveryDateRecord{i + 1}': [dates[i]] for i in range(10)})

    df222 = pd.concat([df111, df_dates111], axis=1)

    temp_df = pd.concat([temp_df, df222], axis=0, ignore_index=True)

my_df2 = pd.concat([my_df2, temp_df], axis=1)

my_df2['PNLatZeroCh'] = my_df2['PNLatZeroCh'].round(2)
my_df2["Max"] = my_df2["Max"].round(2)
my_df2["Min"] = my_df2["Min"].round(2)
my_df2["Mean"] = my_df2["Mean"].round(2)
my_df2["Median"] = my_df2["Median"].round(2)
my_df2["Mean/Min"] = my_df2["Mean/Min"] * -1
my_df2["Mean/Min"] = my_df2["Mean/Min"].round(2)
my_df2["Median/Min"] = my_df2["Median/Min"] * -1
my_df2["Median/Min"] = my_df2["Median/Min"].round(2)
my_df2["LCount"] = my_df2["LCount"].round(2)
my_df2["PCount"] = my_df2["PCount"].round(2)
my_df2["LSum"] = my_df2["LSum"].round(2)
my_df2["PSum"] = my_df2["PSum"].round(2)
my_df2["LMean"] = my_df2["LMean"].round(2)
my_df2["PMean"] = my_df2["PMean"].round(2)
my_df2["PMean/LMean"] = my_df2["PMean/LMean"] * -1
my_df2["PMean/LMean"] = my_df2["PMean/LMean"].round(2)
my_df2["PoP"] = my_df2["PoP"].round(2)
my_df2["Odds"] = my_df2["Odds"].round(2)
my_df2["Rtn"] = my_df2["Rtn"].round(2)
my_df2["RtnAnn"] = my_df2["RtnAnn"].round(2)
my_df2["MaxDD"] = my_df2["MaxDD"].round(2)
my_df2["MeanDaysInRecovery"] = my_df2["MeanDaysInRecovery"].round(2)
my_df2["MaxDaysInLoosingStreak"] = my_df2["MaxDaysInLoosingStreak"].round(2)
my_df2["MeanDaysInLoosingStreakLocalMax"] = my_df2["MeanDaysInLoosingStreakLocalMax"].round(2)
my_df2["Calmar Simplified"] = my_df2["Calmar Simplified"] * -1
my_df2["Calmar Simplified"] = my_df2["Calmar Simplified"].round(2)

print(datetime.now())

combinations = list(product(my_df2['ID2'], my_df1['ID1']))

combined_df = pd.DataFrame({'ID': [', '.join(comb) for comb in combinations]})

combined_df['PnLs'] = [
    [a + b for a, b in zip(my_df2.loc[my_df2['ID2'] == comb[0], 'PnLs'].values[0],
                           my_df1.loc[my_df1['ID1'] == comb[1], 'PnLs'].values[0])]
    for comb in combinations
]

c = 0
i = 0
OPtions_count = []
SOME_PNL = []
for index, row in combined_df.iterrows():
    if i != 0 and i % 149 == 0:
        i = 0
        c += 1
    SOME_PNL.append(PNL_AT_ZERO_PB[c] + PNL_AT_ZERO_CB[i])
    Equity = [Equity_0]
    arr = row['PnLs']
    i_k = 0
    DD = [0]
    DaysinDD = []
    minimum = min(arr) * 100
    maximum = max(arr) * 100
    mean = statistics.mean(arr) * 100
    median = statistics.median(arr) * 100
    p_count, n_count, p_sum, n_sum = count_neg_pos(arr)
    prob_of_profit = p_count/len(arr)
    if n_sum == 0:
        n_sum = 1
    odds = p_sum / abs(n_sum)
    if n_count == 0:
        n_count = 1
    l_mean = (n_sum / n_count) * 100
    if p_count == 0:
        p_count = 1
    p_mean = (p_sum / p_count) * 100

    for ind, pnl in enumerate(arr):
        equity_i = Equity[-1] + (pnl * ((Equity[-1] / TradeSizeFactor) / (prices_p[c] + prices_c[i])))
        Equity.append(equity_i)
        DD.append(Equity[-1] / max(Equity) - 1)
        if DD[-1] != 0:
            i_k += 1
        elif DD[-1] == 0:
            i_k = 0
        DaysinDD.append(i_k)
    i += 1
    DaysInRecoveryForDates = DaysIRecoveryForDates(DaysinDD, df_prices["Date"])
    top_10 = top_n_largest_without_nan(DaysInRecoveryForDates, 10)
    all_top_10.append(top_10)
    DaysInRecovery = DaysIRecovery(DaysinDD)
    if row["ID"] == "P433, C425":
        print(DD)
        print(DaysinDD)
        print(DaysInRecovery)
    MeanDaysInRecovery = statistics.mean(DaysInRecovery)
    DaysInLoosingStreak = DaysILoosingStreak(arr)
    DaysInLoosingStreakLocalMax = DaysILoosingStreakLocalMax(DaysInLoosingStreak)
    MaxDaysInLoosingStreak = max(DaysInLoosingStreak)
    MeanDaysInLoosingStreakLocalMax = statistics.mean(DaysInLoosingStreakLocalMax)
    Rtn = Equity[-1] / Equity[0] - 1
    RtnAnn = (Rtn + 1) ** (252 / len(arr)) - 1
    MaxDD = min(DD)
    if MaxDD == 0:
        Calm_simple = 0
    else:
        Calm_simple = RtnAnn/MaxDD

    OPtions_count.append(2)
    mins.append(minimum)
    maxs.append(maximum)
    means.append(mean)
    l_means.append(l_mean)
    p_means.append(p_mean)
    p_mean_l_means.append(p_mean/l_mean)
    medians.append(median)
    mean_mins.append(mean / minimum)
    median_mins.append(median / minimum)
    p_counts.append(p_count)
    n_counts.append(n_count)
    p_sums.append(p_sum * 100)
    n_sums.append(n_sum * 100)
    prob_of_profits.append(prob_of_profit)
    oddss.append(odds)
    Rtns.append(Rtn)
    RtnAnns.append(RtnAnn)
    Max_DDs.append(MaxDD)
    Calm_simples.append(Calm_simple)
    MaxDaysinDD.append(max(DaysinDD))
    MeanDaysInRecovery_s.append(MeanDaysInRecovery)
    MaxDaysInLoosingStreak_s.append(MaxDaysInLoosingStreak)
    MeanDaysInLoosingStreakLocalMax_s.append(MeanDaysInLoosingStreakLocalMax)

temp_df = pd.DataFrame()

sort_str = []
for ind, row in combined_df.iterrows():
    U = 437
    x = row['ID']
    substrings = x.split(', ')
    result_string = ', '.join([f'{substring[:2]}{int(substring[2:]) - U:+}' for substring in substrings])
    sort_str.append(result_string)

combined_df['Options Count'] = OPtions_count
combined_df['ID2'] = sort_str
combined_df['PNLatZeroCh'] = SOME_PNL
combined_df["Max"] = maxs
combined_df["Min"] = mins
combined_df["Mean"] = means
combined_df["Median"] = medians
combined_df["Mean/Min"] = mean_mins
combined_df["Medain/Min"] = median_mins
combined_df["LCount"] = n_counts
combined_df["PCount"] = p_counts
combined_df["LSum"] = n_sums
combined_df["PSum"] = p_sums
combined_df["LMean"] = l_means
combined_df["PMean"] = p_means
combined_df["PMean/LMean"] = p_mean_l_means
combined_df["PoP"] = prob_of_profits
combined_df["Odds"] = oddss
combined_df["Rtn"] = Rtns
combined_df["RtnAnn"] = RtnAnns
combined_df["MaxDD"] = Max_DDs
combined_df["MaxDaysInDD"] = MaxDaysinDD
combined_df["MeanDaysInRecovery"] = MeanDaysInRecovery_s
combined_df["MaxDaysInLoosingStreak"] = MaxDaysInLoosingStreak_s
combined_df["MeanDaysInLoosingStreakLocalMax"] = MeanDaysInLoosingStreakLocalMax_s
combined_df["Calmar Simplified"] = Calm_simples
for i, top_10 in enumerate(all_top_10):
    numbers = [num for num, date in top_10]
    dates = [date for num, date in top_10]

    df111 = pd.DataFrame({f'DaysInRecoveryRecord{i + 1}': [numbers[i]] for i in range(10)})
    df_dates111 = pd.DataFrame({f'DaysInRecoveryDateRecord{i + 1}': [dates[i]] for i in range(10)})

    df222 = pd.concat([df111, df_dates111], axis=1)

    temp_df = pd.concat([temp_df, df222], axis=0, ignore_index=True)
combined_df = pd.concat([combined_df, temp_df], axis=1)

combined_df["Max"] = combined_df["Max"].round(2)
combined_df["Min"] = combined_df["Min"].round(2)
combined_df["Mean"] = combined_df["Mean"].round(2)
combined_df["Median"] = combined_df["Median"].round(2)
combined_df["Mean/Min"] = combined_df["Mean/Min"] * -1
combined_df["Mean/Min"] = combined_df["Mean/Min"].round(2)
combined_df["Medain/Min"] = combined_df["Medain/Min"] * -1
combined_df["Medain/Min"] = combined_df["Medain/Min"].round(2)
combined_df["LCount"] = combined_df["LCount"].round(2)
combined_df["PCount"] = combined_df["PCount"].round(2)
combined_df["LSum"] = combined_df["LSum"].round(2)
combined_df["PSum"] = combined_df["PSum"].round(2)
combined_df["LMean"] = combined_df["LMean"].round(2)
combined_df["PMean"] = combined_df["PMean"].round(2)
combined_df["PMean/LMean"] = combined_df["PMean/LMean"] * -1
combined_df["PMean/LMean"] = combined_df["PMean/LMean"].round(2)
combined_df["PoP"] = combined_df["PoP"].round(2)
combined_df["Odds"] = combined_df["Odds"].round(2)
combined_df["Rtn"] = combined_df["Rtn"].round(2)
combined_df["RtnAnn"] = combined_df["RtnAnn"].round(2)
combined_df["MaxDD"] = combined_df["MaxDD"].round(2)
combined_df["MeanDaysInRecovery"] = combined_df["MeanDaysInRecovery"].round(2)
combined_df["MaxDaysInLoosingStreak"] = combined_df["MaxDaysInLoosingStreak"].round(2)
combined_df["MeanDaysInLoosingStreakLocalMax"] = combined_df["MeanDaysInLoosingStreakLocalMax"].round(2)
combined_df["Calmar Simplified"] = combined_df["Calmar Simplified"] * -1
combined_df["Calmar Simplified"] = combined_df["Calmar Simplified"].round(2)

print(datetime.now())

num_random_combinations = 1000

result_combinations = []

FINAL_ZERO_PNL = []

while len(result_combinations) < num_random_combinations:
    start = 77
    end = 92
    random_index1 = np.random.randint(start, end)
    random_index2 = np.random.randint(start, end)
    while random_index2 == random_index1:
        random_index2 = np.random.randint(start, end)
    random_index3 = np.random.randint(start, end)
    random_index4 = np.random.randint(start, end)
    while random_index4 == random_index3:
        random_index4 = np.random.randint(start, end)
    p_id1 = my_df2.at[random_index1, 'ID2'][1:]
    p_id2 = my_df2.at[random_index2, 'ID2'][1:]
    c_id1 = my_df1.at[random_index3, 'ID1'][1:]
    c_id2 = my_df1.at[random_index4, 'ID1'][1:]
    put_data1 = PnLss_put.get(p_id1, [])
    put_data2 = PnLss_put1.get(p_id2, [])
    call_data1 = PnLss_call.get(c_id1, [])
    call_data2 = PnLss_call1.get(c_id2, [])
    my_put_data1 = PNL_PB.get(p_id1)
    my_put_data2 = PNL_PS.get(p_id2)
    my_call_data1 = PNL_CB.get(c_id1)
    my_call_data2 = PNL_CS.get(c_id2)

    pnl_sum = [a - b + c - d for a, b, c, d in zip(put_data1, put_data2, call_data1, call_data2)]

    random_combination = [
        f"b{p_id1}",
        f"s{p_id2}",
        f"b{c_id1}",
        f"s{c_id2}",
        pnl_sum
    ]

    if random_combination not in result_combinations:
        result_combinations.append(random_combination)
        FINAL_ZERO_PNL.append(my_put_data1 - my_put_data2 + my_call_data1 - my_call_data2)

merged_df = pd.DataFrame({'ID': [', '.join(map(str, comb[:-1])) for comb in result_combinations],
                          'PNL': [comb[-1] for comb in result_combinations]})

sort_str = []
for ind, row in merged_df.iterrows():
    arr = row['ID']
    substrings = arr.split(', ')
    sorted_substrings = sorted(substrings, key=lambda x: int(x[3:]))
    result_string = ', '.join(sorted_substrings)
    sort_str.append(result_string)

merged_df['ID'] = sort_str


pattern = re.compile(r'(\d+)')
Options_count = []
Mins = []
Maxs = []
Means = []
LMeans = []
PMeans = []
PMeans_LMeans = []
Medians = []
MeanMins = []
MedianMeans = []
PCounts = []
NCounts = []
PSums = []
NSums = []
PRob_of_profits = []
ODdss = []
RTNs = []
RTNANNs = []
MAX_DDs = []
CALM_SIMPLEs = []
MAX_DAYS_IN_DDs = []
MEAN_DAYS_IN_RECOVERYs = []
MAX_DAYS_IN_LOOSING_STREAKs = []
MEAN_DAYS_IN_LOOSING_STREAK_LOCAL_MAX = []
all_top_10_CM = []
REQ = []
for idx, row in merged_df.iterrows():
    my_id = row['ID']
    firsts = re.findall(r'\b\w', my_id)
    arr = row['PNL']
    i_k = 0
    numbers = pattern.findall(my_id)

    first_letters = re.findall(r'\b\w', my_id)
    res = ''.join(first_letters)
    letters = list(res)
    idents = my_id.split(', ')
    prices = []
    for ix, letter in zip(idents, letters):
        letter_part = letter
        some_part = ix[1:]

        if letter == 'b':
            if some_part[0] == 'P':
                val = my_pb_dict.get(some_part)
            elif some_part[0] == 'C':
                val = my_cb_dict.get(some_part)
        elif letter == 's':
            if some_part[0] == 'P':
                val = my_ps_dict.get(some_part)
            elif some_part[0] == 'C':
                val = my_cs_dict.get(some_part)

        prices.append(val)

    Equity = [Equity_0]
    NumberOfContracts_comb = []
    ToAdd = []
    DD = [0]
    DaysinDD = []
    minimum = min(arr) * 100
    maximum = max(arr) * 100
    mean = statistics.mean(arr) * 100
    median = statistics.median(arr) * 100
    p_count, n_count, p_sum, n_sum = count_neg_pos(arr)
    prob_of_profit = p_count / len(arr)
    if n_sum == 0:
        n_sum = 1
    odds = p_sum / abs(n_sum)
    if n_count == 0:
        n_count = 1
    l_mean = (n_sum / n_count) * 100
    if p_count == 0:
        p_count = 1
    p_mean = (p_sum / p_count) * 100

    Trades = []
    strike1, strike2, strike3, strike4 = map(int, numbers)
    TradeUnderlyingPrice01 = strike1 - 1
    trade = []
    price21 = TradeUnderlyingPrice01 - strike1
    if price21 < 0:
        price21 = 0
    price22 = TradeUnderlyingPrice01 - strike2
    if price22 < 0:
        price22 = 0
    price23 = TradeUnderlyingPrice01 - strike3
    if price23 < 0:
        price23 = 0
    price24 = TradeUnderlyingPrice01 - strike4
    if price24 < 0:
        price24 = 0
    trade.append(price21 - prices[0])
    trade.append(price22 - prices[1])
    trade.append(price23 - prices[2])
    trade.append(price24 - prices[3])
    Trades.append(sum(trade))
    TradeUnderlyingPrice12 = strike1 + (strike2 - strike1) / 2
    trade = []
    price21 = TradeUnderlyingPrice12 - strike1
    if price21 < 0:
        price21 = 0
    price22 = TradeUnderlyingPrice12 - strike2
    if price22 < 0:
        price22 = 0
    price23 = TradeUnderlyingPrice12 - strike3
    if price23 < 0:
        price23 = 0
    price24 = TradeUnderlyingPrice12 - strike4
    if price24 < 0:
        price24 = 0
    trade.append(price21 - prices[0])
    trade.append(price22 - prices[1])
    trade.append(price23 - prices[2])
    trade.append(price24 - prices[3])
    Trades.append(sum(trade))
    TradeUnderlyingPrice23 = strike2 + (strike3 - strike2) / 2
    trade = []
    price21 = TradeUnderlyingPrice23 - strike1
    if price21 < 0:
        price21 = 0
    price22 = TradeUnderlyingPrice23 - strike2
    if price22 < 0:
        price22 = 0
    price23 = TradeUnderlyingPrice23 - strike3
    if price23 < 0:
        price23 = 0
    price24 = TradeUnderlyingPrice23 - strike4
    if price24 < 0:
        price24 = 0
    trade.append(price21 - prices[0])
    trade.append(price22 - prices[1])
    trade.append(price23 - prices[2])
    trade.append(price24 - prices[3])
    Trades.append(sum(trade))
    TradeUnderlyingPrice34 = strike3 + (strike4 - strike3) / 2
    trade = []
    price21 = TradeUnderlyingPrice34 - strike1
    if price21 < 0:
        price21 = 0
    price22 = TradeUnderlyingPrice34 - strike2
    if price22 < 0:
        price22 = 0
    price23 = TradeUnderlyingPrice34 - strike3
    if price23 < 0:
        price23 = 0
    price24 = TradeUnderlyingPrice34 - strike4
    if price24 < 0:
        price24 = 0
    trade.append(price21 - prices[0])
    trade.append(price22 - prices[1])
    trade.append(price23 - prices[2])
    trade.append(price24 - prices[3])
    Trades.append(sum(trade))
    TradeUnderlyingPrice4inf = strike4 + 1
    trade = []
    price21 = TradeUnderlyingPrice4inf - strike1
    if price21 < 0:
        price21 = 0
    price22 = TradeUnderlyingPrice4inf - strike2
    if price22 < 0:
        price22 = 0
    price23 = TradeUnderlyingPrice4inf - strike3
    if price23 < 0:
        price23 = 0
    price24 = TradeUnderlyingPrice4inf - strike4
    if price24 < 0:
        price24 = 0
    trade.append(price21 - prices[0])
    trade.append(price22 - prices[1])
    trade.append(price23 - prices[2])
    trade.append(price24 - prices[3])
    Trades.append(sum(trade))

    RequiredCapital = min(Trades)

    REQ.append(RequiredCapital)

    for ind, pnl in enumerate(arr):
        ik = 1

        NumberOfContracts_comb.append(Equity[-1]/TradeSizeFactor/RequiredCapital)
        ToAdd.append(pnl*NumberOfContracts_comb[-1])
        Equity.append(Equity[-1] + ToAdd[-1])
        DD.append(Equity[-1] / max(Equity) - 1)
        if DD[-1] != 0:
            i_k += 1
        elif DD[-1] == 0:
            i_k = 0
        DaysinDD.append(i_k)
    DaysInRecoveryForDates = DaysIRecoveryForDates(DaysinDD, df_prices["Date"])
    top_10 = top_n_largest_without_nan(DaysInRecoveryForDates, 10)
    all_top_10_CM.append(top_10)
    DaysInRecovery = DaysIRecovery(DaysinDD)
    if len(DaysInRecovery) >= 1:
        MeanDaysInRecovery = statistics.mean(DaysInRecovery)
    else:
        MeanDaysInRecovery = float("NaN")
    DaysInLoosingStreak = DaysILoosingStreak(arr)
    DaysInLoosingStreakLocalMax = DaysILoosingStreakLocalMax(DaysInLoosingStreak)
    MaxDaysInLoosingStreak = max(DaysInLoosingStreak)
    if len(DaysInLoosingStreakLocalMax) >= 1:
        MeanDaysInLoosingStreakLocalMax = statistics.mean(DaysInLoosingStreakLocalMax)
    else:
        MeanDaysInLoosingStreakLocalMax = float("NaN")
    Rtn = Equity[-1] / Equity[0] - 1
    RtnAnn = (Rtn + 1) ** (252 / len(arr)) - 1
    MaxDD = min(DD)
    if MaxDD == 0:
        Calm_simple = 0
    else:
        Calm_simple = RtnAnn / MaxDD
    if l_mean == 0:
        agr = float("NaN")
    else:
        agr = p_mean/l_mean

    Options_count.append(4)
    Mins.append(minimum)
    Maxs.append(maximum)
    Means.append(mean)
    LMeans.append(l_mean)
    PMeans.append(p_mean)
    PMeans_LMeans.append(agr)
    Medians.append(median)
    if minimum == 0:
        MeanMins.append(float("NaN"))
        MedianMeans.append(float("NaN"))
    else:
        MeanMins.append(mean/minimum)
        MedianMeans.append(median/minimum)
    PCounts.append(p_count)
    NCounts.append(n_count)
    PSums.append(p_sum * 100)
    NSums.append(n_sum * 100)
    PRob_of_profits.append(prob_of_profit)
    ODdss.append(odds)
    RTNs.append(Rtn)
    RTNANNs.append(RtnAnn)
    MAX_DDs.append(MaxDD)
    CALM_SIMPLEs.append(Calm_simple)
    MAX_DAYS_IN_DDs.append(max(DaysinDD))
    MEAN_DAYS_IN_RECOVERYs.append(MeanDaysInRecovery)
    MAX_DAYS_IN_LOOSING_STREAKs.append(MaxDaysInLoosingStreak)
    MEAN_DAYS_IN_LOOSING_STREAK_LOCAL_MAX.append(MeanDaysInLoosingStreakLocalMax)

temp_df = pd.DataFrame()

sort_sort_str = []
for x in sort_str:
    U = 437
    substrings = x.split(', ')
    result_string = ', '.join([f'{substring[:2]}{int(substring[2:]) - U:+}' for substring in substrings])
    sort_sort_str.append(result_string)

merged_df['Options Count'] = Options_count
merged_df['ID2'] = sort_sort_str
merged_df['PNLatZeroCh'] = FINAL_ZERO_PNL
merged_df["Max"] = Maxs
merged_df["Min"] = Mins
merged_df['Required Capital'] = REQ
merged_df["Mean"] = Means
merged_df["Median"] = Medians
merged_df["Mean/Min"] = MeanMins
merged_df["Medain/Min"] = MedianMeans
merged_df["LCount"] = NCounts
merged_df["PCount"] = PCounts
merged_df["LSum"] = NSums
merged_df["PSum"] = PSums
merged_df["LMean"] = LMeans
merged_df["PMean"] = PMeans
merged_df["PMean/LMean"] = PMeans_LMeans
merged_df["PoP"] = PRob_of_profits
merged_df["Odds"] = ODdss
merged_df["Rtn"] = RTNs
merged_df["RtnAnn"] = RTNANNs
merged_df["MaxDD"] = MAX_DDs
merged_df["MaxDaysInDD"] = MAX_DAYS_IN_DDs
merged_df["MeanDaysInRecovery"] = MEAN_DAYS_IN_RECOVERYs
merged_df["MaxDaysInLoosingStreak"] = MAX_DAYS_IN_LOOSING_STREAKs
merged_df["MeanDaysInLoosingStreakLocalMax"] = MEAN_DAYS_IN_LOOSING_STREAK_LOCAL_MAX
merged_df["Calmar Simplified"] = CALM_SIMPLEs
for i, top_10 in enumerate(all_top_10_CM):
    numbers = [num for num, date in top_10]
    dates = [date for num, date in top_10]

    df111 = pd.DataFrame({f'DaysInRecoveryRecord{i + 1}': [numbers[i]] for i in range(10)})
    df_dates111 = pd.DataFrame({f'DaysInRecoveryDateRecord{i + 1}': [dates[i]] for i in range(10)})

    df222 = pd.concat([df111, df_dates111], axis=1)

    temp_df = pd.concat([temp_df, df222], axis=0, ignore_index=True)
merged_df = pd.concat([merged_df, temp_df], axis=1)

merged_df["PNLatZeroCh"] = merged_df["PNLatZeroCh"].round(2)
merged_df["Max"] = merged_df["Max"].round(2)
merged_df["Min"] = merged_df["Min"].round(2)
merged_df["Required Capital"] = merged_df["Required Capital"].round(2)
merged_df["Mean"] = merged_df["Mean"].round(2)
merged_df["Median"] = merged_df["Median"].round(2)
merged_df["Mean/Min"] = merged_df["Mean/Min"] * -1
merged_df["Mean/Min"] = merged_df["Mean/Min"].round(2)
merged_df["Medain/Min"] = merged_df["Medain/Min"] * -1
merged_df["Medain/Min"] = merged_df["Medain/Min"].round(2)
merged_df["LCount"] = merged_df["LCount"].round(2)
merged_df["PCount"] = merged_df["PCount"].round(2)
merged_df["LSum"] = merged_df["LSum"].round(2)
merged_df["PSum"] = merged_df["PSum"].round(2)
merged_df["LMean"] = merged_df["LMean"].round(2)
merged_df["PMean"] = merged_df["PMean"].round(2)
merged_df["PMean/LMean"] = merged_df["PMean/LMean"] * -1
merged_df["PMean/LMean"] = merged_df["PMean/LMean"].round(2)
merged_df["PoP"] = merged_df["PoP"].round(2)
merged_df["Odds"] = merged_df["Odds"].round(2)
merged_df["Rtn"] = merged_df["Rtn"].round(2)
merged_df["RtnAnn"] = merged_df["RtnAnn"].round(2)
merged_df["MaxDD"] = merged_df["MaxDD"].round(2)
merged_df["MaxDaysInDD"] = merged_df["MaxDaysInDD"].round(2)
merged_df["MeanDaysInRecovery"] = merged_df["MeanDaysInRecovery"].round(2)
merged_df["MaxDaysInLoosingStreak"] = merged_df["MaxDaysInLoosingStreak"].round(2)
merged_df["MeanDaysInLoosingStreakLocalMax"] = merged_df["MeanDaysInLoosingStreakLocalMax"].round(2)
merged_df["Calmar Simplified"] = merged_df["Calmar Simplified"] * -1
merged_df["Calmar Simplified"] = merged_df["Calmar Simplified"].round(2)

print(datetime.now())

underlying_data = pd.DataFrame({'Snaphot Date': ['2023-11-07'], 'Expiration Date': ['2023-11-08'], 'Equity': [10000], 'TradeSizeFactor': [100], 'Underlying price': [436.93], 'First day': ['2022-11-28'], 'Last day': ['2023-11-07']})


combined_df = combined_df.drop(['PnLs'], axis=1)
my_df1 = my_df1.drop(['PnLs'], axis=1)
my_df2 = my_df2.drop(['PnLs'], axis=1)
merged_df = merged_df.drop(['PNL'], axis=1)

underlying_data.to_csv('OPTIONS_PNLs_2023-11-24_V017_SHORT.csv', index=False)
with open('OPTIONS_PNLs_2023-11-24_V017_SHORT.csv', 'a') as file:
    file.write('\n' + '\n')

my_df1.to_csv('OPTIONS_PNLs_2023-11-24_V017_SHORT.csv', mode='a', header=True, index=False)

my_df2.to_csv('OPTIONS_PNLs_2023-11-24_V017_SHORT.csv', mode='a', header=False, index=False)

combined_df.to_csv('OPTIONS_PNLs_2023-11-24_V017_SHORT.csv', mode='a', header=False, index=False)

merged_df.to_csv('OPTIONS_PNLs_2023-11-24_V017_SHORT.csv', mode='a', header=False, index=False)