version.py
==========

_get_version()
--------------

.. autofunction:: casanovo.version._get_version()

casanovo.py
===========

main()
------

.. autofunction:: casanovo.casanovo.main()

dataloaders.py
==============

setup(self, stage=None, annotated=True)
---------------------------------------

.. autofunction:: casanovo.denovo.dataloaders.DeNovoDataModule.setup()

_make_loader(self, dataset)
---------------------------

.. autofunction:: casanovo.denovo.dataloaders.DeNovoDataModule._make_loader()

prepare_batch(batch)
--------------------

.. autofunction:: casanovo.denovo.dataloaders.prepare_batch()

evaluate.py
===========

best_aa_match(orig_seq, pred_seq, aa_dict)
------------------------------------------

.. autofunction:: casanovo.denovo.evaluate.best_aa_match()

find_aa_match_single_pep(orig_seq, pred_seq, aa_dict)
-----------------------------------------------------

.. autofunction:: casanovo.denovo.evaluate.find_aa_match_single_pep()

match_aa(orig_seq, pred_seq, aa_dict, eval_direction = 'best')
--------------------------------------------------------------

.. autofunction:: casanovo.denovo.evaluate.match_aa()

batch_aa_match(pred_pep_seqs, true_pep_seqs, aa_dict, eval_direction = 'best')
------------------------------------------------------------------------------

.. autofunction:: casanovo.denovo.evaluate.batch_aa_match()

calc_eval_metrics(aa_match_binary_list, orig_total_num_aa, pred_total_num_aa)
-----------------------------------------------------------------------------

.. autofunction:: casanovo.denovo.evaluate.calc_eval_metrics()

aa_precision_recall_with_threshold(correct_aa_confidences, all_aa_confidences, num_original_aa, threshold)
----------------------------------------------------------------------------------------------------------

.. autofunction:: casanovo.denovo.evaluate.aa_precision_recall_with_threshold()



