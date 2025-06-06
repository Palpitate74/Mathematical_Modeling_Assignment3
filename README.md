# homework3
2025年同济大学春季学期cxd老师数学建模课程第三次大作业C题


2025年五一杯数学建模比赛C题

## C题 社交媒体平台用户分析问题

近年来，社交媒体平台打造了多元化的线上交流空间和文化圈，深刻影响着人们社交互动与信息获取。博主基于专业知识或兴趣爱好等创作出高质量内容，吸引并获得用户的关注。用户可以随时通过观看、点赞、评论等行为积极参与其中。博主依据平台的推荐机制和用户反馈，调整并提升内容质量，从而提高自身影响力。而用户则通过互动行为，反向影响平台的内容推荐系统。

现某社交媒体平台需深入分析现有用户和博主之间的互动行为关系，来预测用户行为，并优化内容推荐方法。附件1记录了该平台在2024.7.11-2024.7.20之间的数据，包括用户ID、用户行为、博主ID、时间。其中用户行为列中，数字1、2、3分别代表用户对博主发布内容的观看、点赞、评论，4代表关注该博主。时间列代表用户行为发生的时间。需要注意的是，用户点赞、评论和关注的行为均代表用户已观看了内容。此外，用户使用该社交媒体平台的频率和时间不同，若某段时间内附件1中没有记录某用户的行为数据，则代表该时段内用户没有使用该社交媒体平台。附件2中记录了2024.7.22用户进行观看、点赞、评论的行为数据。

假设：
1. 该平台用户和博主数量固定，不存在平台新用户/博主的加入和账号注销行为；
2. 用户和博主的互动关系建立后不再变化，即平台中用户不存在取消点赞、删除评论、取消关注的行为。

请结合附件数据，建立数学模型，解决下列问题。

### 问题1
基于用户与博主历史交互数据（观看、点赞、评论、关注）的统计分析，能够有效揭示用户行为特征，为内容优化和交互提升提供决策依据。根据附件1提供的数据，请建立数学模型，预测各博主在2024.7.21当天新增的关注数，并根据预测结果，在表1中填写当日新增关注数最多的5位博主ID及其对应的新增关注数。

#### 表1 问题1结果

| 排名 | 1  | 2  | 3  | 4  | 5  |
| ---- | -- | -- | -- | -- | -- |
| 博主ID |    |    |    |    |    |
| 新增关注数 |    |    |    |    |    |

### 问题2
附件2提供了2024.7.22当天用户进行观看、点赞、评论的行为数据，结合附件1中用户的历史行为数据，请建立数学模型，预测用户在2024.7.22产生的新关注行为，并将指定用户在2024.7.22新关注的博主ID填入表2。

#### 表2 问题2结果

| 用户ID | U7  | U6749 | U5769 | U14990 | U52010 |
| ------ | --- | ----- | ----- | ------ | ------ |
| 新关注博主ID |     |       |       |        |        |
> 注：若用户在2024.7.22关注多名博主，均填入表2；若用户在2024.7.22未新关注博主，无需填写。

### 问题3
用户与博主之间互动数可视为点赞数、评论数、关注数之和，平台可据此制定合理的推荐方案，为用户推送“量身定制”的内容，增加用户与博主之间的互动。请基于附件1数据，建立数学模型，预测指定用户在2024.7.21当天是否在线（即使用该社交媒体平台），如果在线，进一步预测该用户可能与博主产生的互动关系，并给出可能与其产生互动数最高的3名博主，将对应的博主ID填入表3。

#### 表3 问题3结果

| 用户ID | U9  | U22405 | U16  | U48420 |
| ------ | --- | ------ | ---- | ------ |
| 博主ID 1 |     |        |      |        |
| 博主ID 2 |     |        |      |        |
| 博主ID 3 |     |        |      |        |
> 注：若该用户在2024.7.21未使用该社交媒体平台，则无需填写。

### 问题4
平台在制定推荐方案时，会充分考虑不同用户使用社交媒体的时间习惯。在问题3的基础上，基于附件1数据，建立数学模型，预测表4中指定用户在2024.7.23是否在线（即使用社交媒体平台），进一步预测该用户在每个在线时段与每个博主的互动数，给出该互动数最高的3名博主ID以及对应的时段，并将结果填入表4。

#### 表4 问题4结果

| 用户ID | U10  | U1951 | U1833 | U26447 |
| ------ | ---- | ----- | ----- | ------ |
| 博主ID 1 |      |       |       |        |
| 时段1    |      |       |       |        |
| 博主ID 2 |      |       |       |        |
| 时段2    |      |       |       |        |
| 博主ID 3 |      |       |       |        |
| 时段3    |      |       |       |        |
> 注：若该用户在2024.7.23未使用该社交媒体平台，则无需填写；推荐时段，只能在以下24个时段中选取：0:00-1:00, 1:00-2:00, ……, 23:00-24:00。

## 解题方法
- Q1:RandomForest、MLP、SVR、LinearRegression、Lasso 、高斯过程回归 (GaussianProcessRegressor) 以及 XGBoost，最后高斯过程回归 (GaussianProcessRegressor)与XGBoost表现最佳
- Q2:XGBoost、CatBoost、三层神经网络，最后XGBoost表现最佳
- Q3/Q4:LSTM

### ___注___

数据预处理代码及部分预处理后的数据集在数据预处理.zip中
