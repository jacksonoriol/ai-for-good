def test_load():
  return 'loaded'
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]
def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + 0.01
def cond_probs_product(table, evidence_row, target, target_value):
  evidence_columns = up_drop_column(table, target)
  evidence_columns = up_list_column_names(evidence_columns)
  return up_product([cond_prob(table, e[0], e[1], target, target_value) for e in up_zip_lists(evidence_columns, evidence_row)])
def prior_prob(table, column, value):
  t_list = up_get_column(table, column)
  p_a = sum([1 if v==value else 0 for v in t_list])/len(t_list)
  return p_a
def naive_bayes(table, evidence_row, target):
  neg1 = cond_probs_product(table, evidence_row, target, 0)
  neg = neg1 * prior_prob(table, target, 0)
  pos1=cond_probs_product(table, evidence_row, target, 1)
  pos = pos1 * prior_prob(table, target, 1)
  x,y = compute_probs(neg,pos)
  return [x,y]
def metrics(zipped_list):
  #asserts here
  assert isinstance(zipped_list,list), 'Parameter must be a list'
  assert all([isinstance(item, list) for item in zipped_list]), 'Parameter must be a list of lists'
  assert all([len(item) ==2 for item in zipped_list]), 'Parameter must be a zipped list of pairs'
  assert all([isinstance(a, int) and isinstance(b, int) for a,b in zipped_list]), 'Parameter must contain all integers'
  assert all([ a >=0 and b >=0 for a,b in zipped_list]), 'Parameter must be greater or equal to zero '
  #body of function below
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  precision = tp/(tp+fp) if tp+fp != 0 else 0
  recall = tp/(tp+fn) if tp+fn != 0 else 0
  f1 = 2 * (precision * recall) / (precision + recall) if precision+recall != 0 else 0
  accuracy = sum(1 if pair[0] == pair[1] else 0 for pair in zipped_list)/len(zipped_list)

  new_dict = {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}

  return new_dict
  
def run_random_forest(train, test, target, n):
  X = up_drop_column(train, target)
  y = up_get_column(train, target)  

  k_feature_table = up_drop_column(test, target)
  k_actuals = up_get_column(test, target)  

  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)

  clf.fit(X, y)  #builds the trees as specified above
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for n,p in probs]  #probs is list of [neg,pos] like we are used to seeing.

  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(all_predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  metrics_table = up_metrics_table(all_mets)
  return metrics_table
