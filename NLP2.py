{
 "cells": [
  {
   "cell_type": "markdown",
   "source": [
    "# GuideLines\n",
    "- **The index of the document is all the same**\n",
    "    - ex) document pointed by input_data[0][0] == document pointed by features[0][0] == .. == document pointed by test_data[0][0] == ..\n",
    "         == document pointed by test_all_tfidf[0][0] == document pointed by my_test_data_path[0][0]\n",
    "- For Input Data\n",
    "    - `input_data`, `features`, `noun`, `all_tf`, `all_df`, `all_idf`, `all_tfidf`, `my_input_data_path`\n",
    "- For Test Data\n",
    "    - `test_data`, `test_features`, `test_noun`, `test_all_tf`, `test_all_df`, `test_all_idf`, `test_all_tfidf`, `my_test_data_path`\n",
    "- Top 5000\n",
    "    - `count_keys`\n",
    "\n",
    "---\n",
    "   - Import Modules\n",
    "   - Load data\n",
    "   - create folder\n",
    "   1. 1.1\n",
    "      1. Extract nouns from input data\n",
    "      2. Extraction of 5000 least frequently\n",
    "      3. Calculate TF\n",
    "      4. Calculate IDF\n",
    "      5. Calculate TF-IDF\n",
    "      6. Write files\n",
    "   2. 1.2\n",
    "      1. Extract nouns from input data\n",
    "      2. Extraction of 5000 least frequently\n",
    "      3. Calculate TF\n",
    "      4. Calculate IDF\n",
    "      5. Calculate TF-IDF\n",
    "      6. Write files\n",
    "   3. 1.3\n",
    "      1. Write train features file\n",
    "   4. 1.4\n",
    "      1. Write test features file\n",
    "   5. 2.1\n",
    "      1. Create X, y for training X, y\n",
    "      2. Create X, y for testing X, y\n",
    "      3. Train SVM Model\n",
    "      4. Predict\n",
    "   6. 2.2\n",
    "      1. Get Scores (Precision, Recall, F1-Score)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "# Import modules"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 650,
   "metadata": {
    "collapsed": true,
    "pycharm": {
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "import os\n",
    "from collections import Counter\n",
    "import natsort      # To Sort files in ascending order\n",
    "import numpy as np\n",
    "from sklearn import svm\n",
    "from sklearn.metrics import plot_confusion_matrix\n",
    "import matplotlib.pyplot as plt     # To visualization\n",
    "from sklearn.metrics import precision_score, recall_score, f1_score\n"
   ]
  },
  {
   "cell_type": "markdown",
   "source": [
    "# 1.1\n",
    "-----\n",
    "## Load Data"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 651,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Corpus/Input_Data/child', 'Corpus/Input_Data/culture', 'Corpus/Input_Data/economy', 'Corpus/Input_Data/education', 'Corpus/Input_Data/health', 'Corpus/Input_Data/life', 'Corpus/Input_Data/person', 'Corpus/Input_Data/policy', 'Corpus/Input_Data/society']\n",
      "[['Corpus/Input_Data/child/9_(POS)child_1.txt', 'Corpus/Input_Data/child/9_(POS)child_2.txt', 'Corpus/Input_Data/child/9_(POS)child_3.txt', 'Corpus/Input_Data/child/9_(POS)child_4.txt', 'Corpus/Input_Data/child/9_(POS)child_5.txt', 'Corpus/Input_Data/child/9_(POS)child_6.txt', 'Corpus/Input_Data/child/9_(POS)child_7.txt', 'Corpus/Input_Data/child/9_(POS)child_8.txt', 'Corpus/Input_Data/child/9_(POS)child_9.txt', 'Corpus/Input_Data/child/9_(POS)child_10.txt', 'Corpus/Input_Data/child/9_(POS)child_11.txt', 'Corpus/Input_Data/child/9_(POS)child_12.txt', 'Corpus/Input_Data/child/9_(POS)child_13.txt', 'Corpus/Input_Data/child/9_(POS)child_14.txt', 'Corpus/Input_Data/child/9_(POS)child_15.txt', 'Corpus/Input_Data/child/9_(POS)child_16.txt', 'Corpus/Input_Data/child/9_(POS)child_17.txt', 'Corpus/Input_Data/child/9_(POS)child_18.txt', 'Corpus/Input_Data/child/9_(POS)child_19.txt', 'Corpus/Input_Data/child/9_(POS)child_20.txt', 'Corpus/Input_Data/child/9_(POS)child_21.txt', 'Corpus/Input_Data/child/9_(POS)child_22.txt', 'Corpus/Input_Data/child/9_(POS)child_23.txt', 'Corpus/Input_Data/child/9_(POS)child_24.txt', 'Corpus/Input_Data/child/9_(POS)child_25.txt', 'Corpus/Input_Data/child/9_(POS)child_26.txt', 'Corpus/Input_Data/child/9_(POS)child_27.txt', 'Corpus/Input_Data/child/9_(POS)child_28.txt', 'Corpus/Input_Data/child/9_(POS)child_29.txt', 'Corpus/Input_Data/child/9_(POS)child_30.txt', 'Corpus/Input_Data/child/9_(POS)child_31.txt', 'Corpus/Input_Data/child/9_(POS)child_32.txt', 'Corpus/Input_Data/child/9_(POS)child_33.txt', 'Corpus/Input_Data/child/9_(POS)child_34.txt', 'Corpus/Input_Data/child/9_(POS)child_35.txt', 'Corpus/Input_Data/child/9_(POS)child_36.txt', 'Corpus/Input_Data/child/9_(POS)child_37.txt', 'Corpus/Input_Data/child/9_(POS)child_38.txt', 'Corpus/Input_Data/child/9_(POS)child_39.txt', 'Corpus/Input_Data/child/9_(POS)child_40.txt', 'Corpus/Input_Data/child/9_(POS)child_41.txt', 'Corpus/Input_Data/child/9_(POS)child_42.txt', 'Corpus/Input_Data/child/9_(POS)child_43.txt', 'Corpus/Input_Data/child/9_(POS)child_44.txt', 'Corpus/Input_Data/child/9_(POS)child_45.txt', 'Corpus/Input_Data/child/9_(POS)child_46.txt', 'Corpus/Input_Data/child/9_(POS)child_47.txt', 'Corpus/Input_Data/child/9_(POS)child_48.txt', 'Corpus/Input_Data/child/9_(POS)child_49.txt', 'Corpus/Input_Data/child/9_(POS)child_50.txt', 'Corpus/Input_Data/child/9_(POS)child_51.txt', 'Corpus/Input_Data/child/9_(POS)child_52.txt', 'Corpus/Input_Data/child/9_(POS)child_53.txt', 'Corpus/Input_Data/child/9_(POS)child_54.txt', 'Corpus/Input_Data/child/9_(POS)child_55.txt', 'Corpus/Input_Data/child/9_(POS)child_56.txt', 'Corpus/Input_Data/child/9_(POS)child_57.txt', 'Corpus/Input_Data/child/9_(POS)child_58.txt', 'Corpus/Input_Data/child/9_(POS)child_59.txt', 'Corpus/Input_Data/child/9_(POS)child_60.txt', 'Corpus/Input_Data/child/9_(POS)child_61.txt', 'Corpus/Input_Data/child/9_(POS)child_62.txt', 'Corpus/Input_Data/child/9_(POS)child_63.txt', 'Corpus/Input_Data/child/9_(POS)child_64.txt', 'Corpus/Input_Data/child/9_(POS)child_65.txt', 'Corpus/Input_Data/child/9_(POS)child_66.txt', 'Corpus/Input_Data/child/9_(POS)child_67.txt', 'Corpus/Input_Data/child/9_(POS)child_68.txt', 'Corpus/Input_Data/child/9_(POS)child_69.txt', 'Corpus/Input_Data/child/9_(POS)child_70.txt', 'Corpus/Input_Data/child/9_(POS)child_71.txt', 'Corpus/Input_Data/child/9_(POS)child_72.txt', 'Corpus/Input_Data/child/9_(POS)child_73.txt', 'Corpus/Input_Data/child/9_(POS)child_74.txt', 'Corpus/Input_Data/child/9_(POS)child_75.txt', 'Corpus/Input_Data/child/9_(POS)child_76.txt', 'Corpus/Input_Data/child/9_(POS)child_77.txt', 'Corpus/Input_Data/child/9_(POS)child_78.txt', 'Corpus/Input_Data/child/9_(POS)child_79.txt', 'Corpus/Input_Data/child/9_(POS)child_80.txt', 'Corpus/Input_Data/child/9_(POS)child_81.txt', 'Corpus/Input_Data/child/9_(POS)child_82.txt', 'Corpus/Input_Data/child/9_(POS)child_83.txt', 'Corpus/Input_Data/child/9_(POS)child_84.txt', 'Corpus/Input_Data/child/9_(POS)child_85.txt', 'Corpus/Input_Data/child/9_(POS)child_86.txt', 'Corpus/Input_Data/child/9_(POS)child_87.txt', 'Corpus/Input_Data/child/9_(POS)child_88.txt', 'Corpus/Input_Data/child/9_(POS)child_89.txt', 'Corpus/Input_Data/child/9_(POS)child_90.txt', 'Corpus/Input_Data/child/9_(POS)child_91.txt', 'Corpus/Input_Data/child/9_(POS)child_92.txt', 'Corpus/Input_Data/child/9_(POS)child_93.txt', 'Corpus/Input_Data/child/9_(POS)child_94.txt', 'Corpus/Input_Data/child/9_(POS)child_95.txt', 'Corpus/Input_Data/child/9_(POS)child_96.txt', 'Corpus/Input_Data/child/9_(POS)child_97.txt', 'Corpus/Input_Data/child/9_(POS)child_98.txt', 'Corpus/Input_Data/child/9_(POS)child_99.txt', 'Corpus/Input_Data/child/9_(POS)child_100.txt', 'Corpus/Input_Data/child/9_(POS)child_101.txt', 'Corpus/Input_Data/child/9_(POS)child_102.txt', 'Corpus/Input_Data/child/9_(POS)child_103.txt', 'Corpus/Input_Data/child/9_(POS)child_104.txt', 'Corpus/Input_Data/child/9_(POS)child_105.txt', 'Corpus/Input_Data/child/9_(POS)child_106.txt', 'Corpus/Input_Data/child/9_(POS)child_107.txt', 'Corpus/Input_Data/child/9_(POS)child_108.txt', 'Corpus/Input_Data/child/9_(POS)child_109.txt', 'Corpus/Input_Data/child/9_(POS)child_110.txt', 'Corpus/Input_Data/child/9_(POS)child_111.txt', 'Corpus/Input_Data/child/9_(POS)child_112.txt', 'Corpus/Input_Data/child/9_(POS)child_113.txt', 'Corpus/Input_Data/child/9_(POS)child_114.txt', 'Corpus/Input_Data/child/9_(POS)child_115.txt', 'Corpus/Input_Data/child/9_(POS)child_116.txt', 'Corpus/Input_Data/child/9_(POS)child_117.txt', 'Corpus/Input_Data/child/9_(POS)child_118.txt', 'Corpus/Input_Data/child/9_(POS)child_119.txt', 'Corpus/Input_Data/child/9_(POS)child_120.txt', 'Corpus/Input_Data/child/9_(POS)child_121.txt', 'Corpus/Input_Data/child/9_(POS)child_122.txt', 'Corpus/Input_Data/child/9_(POS)child_123.txt', 'Corpus/Input_Data/child/9_(POS)child_124.txt', 'Corpus/Input_Data/child/9_(POS)child_125.txt', 'Corpus/Input_Data/child/9_(POS)child_126.txt', 'Corpus/Input_Data/child/9_(POS)child_127.txt', 'Corpus/Input_Data/child/9_(POS)child_128.txt'], ['Corpus/Input_Data/culture/9_(POS)culture_1.txt', 'Corpus/Input_Data/culture/9_(POS)culture_2.txt', 'Corpus/Input_Data/culture/9_(POS)culture_3.txt', 'Corpus/Input_Data/culture/9_(POS)culture_4.txt', 'Corpus/Input_Data/culture/9_(POS)culture_5.txt', 'Corpus/Input_Data/culture/9_(POS)culture_6.txt', 'Corpus/Input_Data/culture/9_(POS)culture_7.txt', 'Corpus/Input_Data/culture/9_(POS)culture_8.txt', 'Corpus/Input_Data/culture/9_(POS)culture_9.txt', 'Corpus/Input_Data/culture/9_(POS)culture_10.txt', 'Corpus/Input_Data/culture/9_(POS)culture_11.txt', 'Corpus/Input_Data/culture/9_(POS)culture_12.txt', 'Corpus/Input_Data/culture/9_(POS)culture_13.txt', 'Corpus/Input_Data/culture/9_(POS)culture_14.txt', 'Corpus/Input_Data/culture/9_(POS)culture_15.txt', 'Corpus/Input_Data/culture/9_(POS)culture_16.txt', 'Corpus/Input_Data/culture/9_(POS)culture_17.txt', 'Corpus/Input_Data/culture/9_(POS)culture_18.txt', 'Corpus/Input_Data/culture/9_(POS)culture_19.txt', 'Corpus/Input_Data/culture/9_(POS)culture_20.txt', 'Corpus/Input_Data/culture/9_(POS)culture_21.txt', 'Corpus/Input_Data/culture/9_(POS)culture_22.txt', 'Corpus/Input_Data/culture/9_(POS)culture_23.txt', 'Corpus/Input_Data/culture/9_(POS)culture_24.txt', 'Corpus/Input_Data/culture/9_(POS)culture_25.txt', 'Corpus/Input_Data/culture/9_(POS)culture_26.txt', 'Corpus/Input_Data/culture/9_(POS)culture_27.txt', 'Corpus/Input_Data/culture/9_(POS)culture_28.txt', 'Corpus/Input_Data/culture/9_(POS)culture_29.txt', 'Corpus/Input_Data/culture/9_(POS)culture_30.txt', 'Corpus/Input_Data/culture/9_(POS)culture_31.txt', 'Corpus/Input_Data/culture/9_(POS)culture_32.txt', 'Corpus/Input_Data/culture/9_(POS)culture_33.txt', 'Corpus/Input_Data/culture/9_(POS)culture_34.txt', 'Corpus/Input_Data/culture/9_(POS)culture_35.txt', 'Corpus/Input_Data/culture/9_(POS)culture_36.txt', 'Corpus/Input_Data/culture/9_(POS)culture_37.txt', 'Corpus/Input_Data/culture/9_(POS)culture_38.txt', 'Corpus/Input_Data/culture/9_(POS)culture_39.txt', 'Corpus/Input_Data/culture/9_(POS)culture_40.txt', 'Corpus/Input_Data/culture/9_(POS)culture_41.txt', 'Corpus/Input_Data/culture/9_(POS)culture_42.txt', 'Corpus/Input_Data/culture/9_(POS)culture_43.txt', 'Corpus/Input_Data/culture/9_(POS)culture_44.txt', 'Corpus/Input_Data/culture/9_(POS)culture_45.txt', 'Corpus/Input_Data/culture/9_(POS)culture_46.txt', 'Corpus/Input_Data/culture/9_(POS)culture_47.txt', 'Corpus/Input_Data/culture/9_(POS)culture_48.txt', 'Corpus/Input_Data/culture/9_(POS)culture_49.txt', 'Corpus/Input_Data/culture/9_(POS)culture_50.txt', 'Corpus/Input_Data/culture/9_(POS)culture_51.txt', 'Corpus/Input_Data/culture/9_(POS)culture_52.txt', 'Corpus/Input_Data/culture/9_(POS)culture_53.txt', 'Corpus/Input_Data/culture/9_(POS)culture_54.txt', 'Corpus/Input_Data/culture/9_(POS)culture_55.txt', 'Corpus/Input_Data/culture/9_(POS)culture_56.txt', 'Corpus/Input_Data/culture/9_(POS)culture_57.txt', 'Corpus/Input_Data/culture/9_(POS)culture_58.txt', 'Corpus/Input_Data/culture/9_(POS)culture_59.txt', 'Corpus/Input_Data/culture/9_(POS)culture_60.txt', 'Corpus/Input_Data/culture/9_(POS)culture_61.txt', 'Corpus/Input_Data/culture/9_(POS)culture_62.txt', 'Corpus/Input_Data/culture/9_(POS)culture_63.txt', 'Corpus/Input_Data/culture/9_(POS)culture_64.txt', 'Corpus/Input_Data/culture/9_(POS)culture_65.txt', 'Corpus/Input_Data/culture/9_(POS)culture_66.txt', 'Corpus/Input_Data/culture/9_(POS)culture_67.txt', 'Corpus/Input_Data/culture/9_(POS)culture_68.txt', 'Corpus/Input_Data/culture/9_(POS)culture_69.txt', 'Corpus/Input_Data/culture/9_(POS)culture_70.txt', 'Corpus/Input_Data/culture/9_(POS)culture_71.txt', 'Corpus/Input_Data/culture/9_(POS)culture_72.txt', 'Corpus/Input_Data/culture/9_(POS)culture_73.txt', 'Corpus/Input_Data/culture/9_(POS)culture_74.txt', 'Corpus/Input_Data/culture/9_(POS)culture_75.txt', 'Corpus/Input_Data/culture/9_(POS)culture_76.txt', 'Corpus/Input_Data/culture/9_(POS)culture_77.txt', 'Corpus/Input_Data/culture/9_(POS)culture_78.txt', 'Corpus/Input_Data/culture/9_(POS)culture_79.txt', 'Corpus/Input_Data/culture/9_(POS)culture_80.txt', 'Corpus/Input_Data/culture/9_(POS)culture_81.txt', 'Corpus/Input_Data/culture/9_(POS)culture_82.txt', 'Corpus/Input_Data/culture/9_(POS)culture_83.txt', 'Corpus/Input_Data/culture/9_(POS)culture_84.txt', 'Corpus/Input_Data/culture/9_(POS)culture_85.txt', 'Corpus/Input_Data/culture/9_(POS)culture_86.txt', 'Corpus/Input_Data/culture/9_(POS)culture_87.txt', 'Corpus/Input_Data/culture/9_(POS)culture_88.txt', 'Corpus/Input_Data/culture/9_(POS)culture_89.txt', 'Corpus/Input_Data/culture/9_(POS)culture_90.txt', 'Corpus/Input_Data/culture/9_(POS)culture_91.txt', 'Corpus/Input_Data/culture/9_(POS)culture_92.txt', 'Corpus/Input_Data/culture/9_(POS)culture_93.txt', 'Corpus/Input_Data/culture/9_(POS)culture_94.txt', 'Corpus/Input_Data/culture/9_(POS)culture_95.txt', 'Corpus/Input_Data/culture/9_(POS)culture_96.txt', 'Corpus/Input_Data/culture/9_(POS)culture_97.txt', 'Corpus/Input_Data/culture/9_(POS)culture_98.txt', 'Corpus/Input_Data/culture/9_(POS)culture_99.txt', 'Corpus/Input_Data/culture/9_(POS)culture_100.txt', 'Corpus/Input_Data/culture/9_(POS)culture_101.txt', 'Corpus/Input_Data/culture/9_(POS)culture_102.txt', 'Corpus/Input_Data/culture/9_(POS)culture_103.txt', 'Corpus/Input_Data/culture/9_(POS)culture_104.txt', 'Corpus/Input_Data/culture/9_(POS)culture_105.txt', 'Corpus/Input_Data/culture/9_(POS)culture_106.txt', 'Corpus/Input_Data/culture/9_(POS)culture_107.txt', 'Corpus/Input_Data/culture/9_(POS)culture_108.txt', 'Corpus/Input_Data/culture/9_(POS)culture_109.txt', 'Corpus/Input_Data/culture/9_(POS)culture_110.txt', 'Corpus/Input_Data/culture/9_(POS)culture_111.txt', 'Corpus/Input_Data/culture/9_(POS)culture_112.txt', 'Corpus/Input_Data/culture/9_(POS)culture_113.txt', 'Corpus/Input_Data/culture/9_(POS)culture_114.txt', 'Corpus/Input_Data/culture/9_(POS)culture_115.txt', 'Corpus/Input_Data/culture/9_(POS)culture_116.txt', 'Corpus/Input_Data/culture/9_(POS)culture_117.txt', 'Corpus/Input_Data/culture/9_(POS)culture_118.txt', 'Corpus/Input_Data/culture/9_(POS)culture_119.txt', 'Corpus/Input_Data/culture/9_(POS)culture_120.txt', 'Corpus/Input_Data/culture/9_(POS)culture_121.txt', 'Corpus/Input_Data/culture/9_(POS)culture_122.txt', 'Corpus/Input_Data/culture/9_(POS)culture_123.txt', 'Corpus/Input_Data/culture/9_(POS)culture_124.txt', 'Corpus/Input_Data/culture/9_(POS)culture_125.txt', 'Corpus/Input_Data/culture/9_(POS)culture_126.txt', 'Corpus/Input_Data/culture/9_(POS)culture_127.txt', 'Corpus/Input_Data/culture/9_(POS)culture_128.txt', 'Corpus/Input_Data/culture/9_(POS)culture_129.txt', 'Corpus/Input_Data/culture/9_(POS)culture_130.txt', 'Corpus/Input_Data/culture/9_(POS)culture_131.txt', 'Corpus/Input_Data/culture/9_(POS)culture_132.txt', 'Corpus/Input_Data/culture/9_(POS)culture_133.txt', 'Corpus/Input_Data/culture/9_(POS)culture_134.txt', 'Corpus/Input_Data/culture/9_(POS)culture_135.txt', 'Corpus/Input_Data/culture/9_(POS)culture_136.txt', 'Corpus/Input_Data/culture/9_(POS)culture_137.txt', 'Corpus/Input_Data/culture/9_(POS)culture_138.txt', 'Corpus/Input_Data/culture/9_(POS)culture_139.txt', 'Corpus/Input_Data/culture/9_(POS)culture_140.txt', 'Corpus/Input_Data/culture/9_(POS)culture_141.txt', 'Corpus/Input_Data/culture/9_(POS)culture_142.txt', 'Corpus/Input_Data/culture/9_(POS)culture_143.txt', 'Corpus/Input_Data/culture/9_(POS)culture_144.txt', 'Corpus/Input_Data/culture/9_(POS)culture_145.txt', 'Corpus/Input_Data/culture/9_(POS)culture_146.txt', 'Corpus/Input_Data/culture/9_(POS)culture_147.txt', 'Corpus/Input_Data/culture/9_(POS)culture_148.txt', 'Corpus/Input_Data/culture/9_(POS)culture_149.txt', 'Corpus/Input_Data/culture/9_(POS)culture_150.txt', 'Corpus/Input_Data/culture/9_(POS)culture_151.txt', 'Corpus/Input_Data/culture/9_(POS)culture_152.txt', 'Corpus/Input_Data/culture/9_(POS)culture_153.txt', 'Corpus/Input_Data/culture/9_(POS)culture_154.txt', 'Corpus/Input_Data/culture/9_(POS)culture_155.txt', 'Corpus/Input_Data/culture/9_(POS)culture_156.txt', 'Corpus/Input_Data/culture/9_(POS)culture_157.txt', 'Corpus/Input_Data/culture/9_(POS)culture_158.txt', 'Corpus/Input_Data/culture/9_(POS)culture_159.txt', 'Corpus/Input_Data/culture/9_(POS)culture_160.txt', 'Corpus/Input_Data/culture/9_(POS)culture_161.txt', 'Corpus/Input_Data/culture/9_(POS)culture_162.txt', 'Corpus/Input_Data/culture/9_(POS)culture_163.txt', 'Corpus/Input_Data/culture/9_(POS)culture_164.txt', 'Corpus/Input_Data/culture/9_(POS)culture_165.txt', 'Corpus/Input_Data/culture/9_(POS)culture_166.txt', 'Corpus/Input_Data/culture/9_(POS)culture_167.txt', 'Corpus/Input_Data/culture/9_(POS)culture_168.txt', 'Corpus/Input_Data/culture/9_(POS)culture_169.txt', 'Corpus/Input_Data/culture/9_(POS)culture_170.txt', 'Corpus/Input_Data/culture/9_(POS)culture_171.txt', 'Corpus/Input_Data/culture/9_(POS)culture_172.txt', 'Corpus/Input_Data/culture/9_(POS)culture_173.txt', 'Corpus/Input_Data/culture/9_(POS)culture_174.txt', 'Corpus/Input_Data/culture/9_(POS)culture_175.txt', 'Corpus/Input_Data/culture/9_(POS)culture_176.txt', 'Corpus/Input_Data/culture/9_(POS)culture_177.txt', 'Corpus/Input_Data/culture/9_(POS)culture_178.txt', 'Corpus/Input_Data/culture/9_(POS)culture_179.txt', 'Corpus/Input_Data/culture/9_(POS)culture_180.txt', 'Corpus/Input_Data/culture/9_(POS)culture_181.txt', 'Corpus/Input_Data/culture/9_(POS)culture_182.txt', 'Corpus/Input_Data/culture/9_(POS)culture_183.txt', 'Corpus/Input_Data/culture/9_(POS)culture_184.txt', 'Corpus/Input_Data/culture/9_(POS)culture_185.txt', 'Corpus/Input_Data/culture/9_(POS)culture_186.txt', 'Corpus/Input_Data/culture/9_(POS)culture_187.txt', 'Corpus/Input_Data/culture/9_(POS)culture_188.txt', 'Corpus/Input_Data/culture/9_(POS)culture_189.txt', 'Corpus/Input_Data/culture/9_(POS)culture_190.txt', 'Corpus/Input_Data/culture/9_(POS)culture_191.txt', 'Corpus/Input_Data/culture/9_(POS)culture_192.txt', 'Corpus/Input_Data/culture/9_(POS)culture_193.txt', 'Corpus/Input_Data/culture/9_(POS)culture_194.txt', 'Corpus/Input_Data/culture/9_(POS)culture_195.txt', 'Corpus/Input_Data/culture/9_(POS)culture_196.txt', 'Corpus/Input_Data/culture/9_(POS)culture_197.txt', 'Corpus/Input_Data/culture/9_(POS)culture_198.txt', 'Corpus/Input_Data/culture/9_(POS)culture_199.txt', 'Corpus/Input_Data/culture/9_(POS)culture_200.txt', 'Corpus/Input_Data/culture/9_(POS)culture_201.txt', 'Corpus/Input_Data/culture/9_(POS)culture_202.txt', 'Corpus/Input_Data/culture/9_(POS)culture_203.txt', 'Corpus/Input_Data/culture/9_(POS)culture_204.txt', 'Corpus/Input_Data/culture/9_(POS)culture_205.txt', 'Corpus/Input_Data/culture/9_(POS)culture_206.txt', 'Corpus/Input_Data/culture/9_(POS)culture_207.txt', 'Corpus/Input_Data/culture/9_(POS)culture_208.txt', 'Corpus/Input_Data/culture/9_(POS)culture_209.txt', 'Corpus/Input_Data/culture/9_(POS)culture_210.txt', 'Corpus/Input_Data/culture/9_(POS)culture_211.txt', 'Corpus/Input_Data/culture/9_(POS)culture_212.txt', 'Corpus/Input_Data/culture/9_(POS)culture_213.txt', 'Corpus/Input_Data/culture/9_(POS)culture_214.txt', 'Corpus/Input_Data/culture/9_(POS)culture_215.txt', 'Corpus/Input_Data/culture/9_(POS)culture_216.txt', 'Corpus/Input_Data/culture/9_(POS)culture_217.txt', 'Corpus/Input_Data/culture/9_(POS)culture_218.txt', 'Corpus/Input_Data/culture/9_(POS)culture_219.txt'], ['Corpus/Input_Data/economy/9_(POS)economy_1.txt', 'Corpus/Input_Data/economy/9_(POS)economy_2.txt', 'Corpus/Input_Data/economy/9_(POS)economy_3.txt', 'Corpus/Input_Data/economy/9_(POS)economy_4.txt', 'Corpus/Input_Data/economy/9_(POS)economy_5.txt', 'Corpus/Input_Data/economy/9_(POS)economy_6.txt', 'Corpus/Input_Data/economy/9_(POS)economy_7.txt', 'Corpus/Input_Data/economy/9_(POS)economy_8.txt', 'Corpus/Input_Data/economy/9_(POS)economy_9.txt', 'Corpus/Input_Data/economy/9_(POS)economy_10.txt', 'Corpus/Input_Data/economy/9_(POS)economy_11.txt', 'Corpus/Input_Data/economy/9_(POS)economy_12.txt', 'Corpus/Input_Data/economy/9_(POS)economy_13.txt', 'Corpus/Input_Data/economy/9_(POS)economy_14.txt', 'Corpus/Input_Data/economy/9_(POS)economy_15.txt', 'Corpus/Input_Data/economy/9_(POS)economy_16.txt', 'Corpus/Input_Data/economy/9_(POS)economy_17.txt', 'Corpus/Input_Data/economy/9_(POS)economy_18.txt', 'Corpus/Input_Data/economy/9_(POS)economy_19.txt', 'Corpus/Input_Data/economy/9_(POS)economy_20.txt', 'Corpus/Input_Data/economy/9_(POS)economy_21.txt', 'Corpus/Input_Data/economy/9_(POS)economy_22.txt', 'Corpus/Input_Data/economy/9_(POS)economy_23.txt', 'Corpus/Input_Data/economy/9_(POS)economy_24.txt', 'Corpus/Input_Data/economy/9_(POS)economy_25.txt', 'Corpus/Input_Data/economy/9_(POS)economy_26.txt', 'Corpus/Input_Data/economy/9_(POS)economy_27.txt', 'Corpus/Input_Data/economy/9_(POS)economy_28.txt', 'Corpus/Input_Data/economy/9_(POS)economy_29.txt', 'Corpus/Input_Data/economy/9_(POS)economy_30.txt', 'Corpus/Input_Data/economy/9_(POS)economy_31.txt', 'Corpus/Input_Data/economy/9_(POS)economy_32.txt', 'Corpus/Input_Data/economy/9_(POS)economy_33.txt', 'Corpus/Input_Data/economy/9_(POS)economy_34.txt', 'Corpus/Input_Data/economy/9_(POS)economy_35.txt', 'Corpus/Input_Data/economy/9_(POS)economy_36.txt', 'Corpus/Input_Data/economy/9_(POS)economy_37.txt', 'Corpus/Input_Data/economy/9_(POS)economy_38.txt', 'Corpus/Input_Data/economy/9_(POS)economy_39.txt', 'Corpus/Input_Data/economy/9_(POS)economy_40.txt', 'Corpus/Input_Data/economy/9_(POS)economy_41.txt', 'Corpus/Input_Data/economy/9_(POS)economy_42.txt', 'Corpus/Input_Data/economy/9_(POS)economy_43.txt', 'Corpus/Input_Data/economy/9_(POS)economy_44.txt', 'Corpus/Input_Data/economy/9_(POS)economy_45.txt', 'Corpus/Input_Data/economy/9_(POS)economy_46.txt', 'Corpus/Input_Data/economy/9_(POS)economy_47.txt', 'Corpus/Input_Data/economy/9_(POS)economy_48.txt', 'Corpus/Input_Data/economy/9_(POS)economy_49.txt', 'Corpus/Input_Data/economy/9_(POS)economy_50.txt', 'Corpus/Input_Data/economy/9_(POS)economy_51.txt', 'Corpus/Input_Data/economy/9_(POS)economy_52.txt', 'Corpus/Input_Data/economy/9_(POS)economy_53.txt', 'Corpus/Input_Data/economy/9_(POS)economy_54.txt', 'Corpus/Input_Data/economy/9_(POS)economy_55.txt', 'Corpus/Input_Data/economy/9_(POS)economy_56.txt', 'Corpus/Input_Data/economy/9_(POS)economy_57.txt', 'Corpus/Input_Data/economy/9_(POS)economy_58.txt', 'Corpus/Input_Data/economy/9_(POS)economy_59.txt', 'Corpus/Input_Data/economy/9_(POS)economy_60.txt', 'Corpus/Input_Data/economy/9_(POS)economy_61.txt', 'Corpus/Input_Data/economy/9_(POS)economy_62.txt', 'Corpus/Input_Data/economy/9_(POS)economy_63.txt', 'Corpus/Input_Data/economy/9_(POS)economy_64.txt', 'Corpus/Input_Data/economy/9_(POS)economy_65.txt', 'Corpus/Input_Data/economy/9_(POS)economy_66.txt', 'Corpus/Input_Data/economy/9_(POS)economy_67.txt', 'Corpus/Input_Data/economy/9_(POS)economy_68.txt', 'Corpus/Input_Data/economy/9_(POS)economy_69.txt', 'Corpus/Input_Data/economy/9_(POS)economy_70.txt', 'Corpus/Input_Data/economy/9_(POS)economy_71.txt', 'Corpus/Input_Data/economy/9_(POS)economy_72.txt', 'Corpus/Input_Data/economy/9_(POS)economy_73.txt', 'Corpus/Input_Data/economy/9_(POS)economy_74.txt', 'Corpus/Input_Data/economy/9_(POS)economy_75.txt', 'Corpus/Input_Data/economy/9_(POS)economy_76.txt', 'Corpus/Input_Data/economy/9_(POS)economy_77.txt', 'Corpus/Input_Data/economy/9_(POS)economy_78.txt', 'Corpus/Input_Data/economy/9_(POS)economy_79.txt', 'Corpus/Input_Data/economy/9_(POS)economy_80.txt', 'Corpus/Input_Data/economy/9_(POS)economy_81.txt', 'Corpus/Input_Data/economy/9_(POS)economy_82.txt', 'Corpus/Input_Data/economy/9_(POS)economy_83.txt', 'Corpus/Input_Data/economy/9_(POS)economy_84.txt', 'Corpus/Input_Data/economy/9_(POS)economy_85.txt', 'Corpus/Input_Data/economy/9_(POS)economy_86.txt', 'Corpus/Input_Data/economy/9_(POS)economy_87.txt', 'Corpus/Input_Data/economy/9_(POS)economy_88.txt', 'Corpus/Input_Data/economy/9_(POS)economy_89.txt', 'Corpus/Input_Data/economy/9_(POS)economy_90.txt', 'Corpus/Input_Data/economy/9_(POS)economy_91.txt', 'Corpus/Input_Data/economy/9_(POS)economy_92.txt', 'Corpus/Input_Data/economy/9_(POS)economy_93.txt', 'Corpus/Input_Data/economy/9_(POS)economy_94.txt', 'Corpus/Input_Data/economy/9_(POS)economy_95.txt', 'Corpus/Input_Data/economy/9_(POS)economy_96.txt', 'Corpus/Input_Data/economy/9_(POS)economy_97.txt', 'Corpus/Input_Data/economy/9_(POS)economy_98.txt', 'Corpus/Input_Data/economy/9_(POS)economy_99.txt', 'Corpus/Input_Data/economy/9_(POS)economy_100.txt', 'Corpus/Input_Data/economy/9_(POS)economy_101.txt', 'Corpus/Input_Data/economy/9_(POS)economy_102.txt', 'Corpus/Input_Data/economy/9_(POS)economy_103.txt', 'Corpus/Input_Data/economy/9_(POS)economy_104.txt', 'Corpus/Input_Data/economy/9_(POS)economy_105.txt', 'Corpus/Input_Data/economy/9_(POS)economy_106.txt', 'Corpus/Input_Data/economy/9_(POS)economy_107.txt', 'Corpus/Input_Data/economy/9_(POS)economy_108.txt', 'Corpus/Input_Data/economy/9_(POS)economy_109.txt', 'Corpus/Input_Data/economy/9_(POS)economy_110.txt', 'Corpus/Input_Data/economy/9_(POS)economy_111.txt', 'Corpus/Input_Data/economy/9_(POS)economy_112.txt', 'Corpus/Input_Data/economy/9_(POS)economy_113.txt', 'Corpus/Input_Data/economy/9_(POS)economy_114.txt', 'Corpus/Input_Data/economy/9_(POS)economy_115.txt', 'Corpus/Input_Data/economy/9_(POS)economy_116.txt', 'Corpus/Input_Data/economy/9_(POS)economy_117.txt', 'Corpus/Input_Data/economy/9_(POS)economy_118.txt', 'Corpus/Input_Data/economy/9_(POS)economy_119.txt', 'Corpus/Input_Data/economy/9_(POS)economy_120.txt', 'Corpus/Input_Data/economy/9_(POS)economy_121.txt', 'Corpus/Input_Data/economy/9_(POS)economy_122.txt', 'Corpus/Input_Data/economy/9_(POS)economy_123.txt', 'Corpus/Input_Data/economy/9_(POS)economy_124.txt', 'Corpus/Input_Data/economy/9_(POS)economy_125.txt', 'Corpus/Input_Data/economy/9_(POS)economy_126.txt', 'Corpus/Input_Data/economy/9_(POS)economy_127.txt', 'Corpus/Input_Data/economy/9_(POS)economy_128.txt', 'Corpus/Input_Data/economy/9_(POS)economy_129.txt', 'Corpus/Input_Data/economy/9_(POS)economy_130.txt', 'Corpus/Input_Data/economy/9_(POS)economy_131.txt', 'Corpus/Input_Data/economy/9_(POS)economy_132.txt', 'Corpus/Input_Data/economy/9_(POS)economy_133.txt', 'Corpus/Input_Data/economy/9_(POS)economy_134.txt', 'Corpus/Input_Data/economy/9_(POS)economy_135.txt', 'Corpus/Input_Data/economy/9_(POS)economy_136.txt', 'Corpus/Input_Data/economy/9_(POS)economy_137.txt', 'Corpus/Input_Data/economy/9_(POS)economy_138.txt', 'Corpus/Input_Data/economy/9_(POS)economy_139.txt', 'Corpus/Input_Data/economy/9_(POS)economy_140.txt', 'Corpus/Input_Data/economy/9_(POS)economy_141.txt', 'Corpus/Input_Data/economy/9_(POS)economy_142.txt', 'Corpus/Input_Data/economy/9_(POS)economy_143.txt', 'Corpus/Input_Data/economy/9_(POS)economy_144.txt', 'Corpus/Input_Data/economy/9_(POS)economy_145.txt', 'Corpus/Input_Data/economy/9_(POS)economy_146.txt', 'Corpus/Input_Data/economy/9_(POS)economy_147.txt', 'Corpus/Input_Data/economy/9_(POS)economy_148.txt', 'Corpus/Input_Data/economy/9_(POS)economy_149.txt', 'Corpus/Input_Data/economy/9_(POS)economy_150.txt', 'Corpus/Input_Data/economy/9_(POS)economy_151.txt', 'Corpus/Input_Data/economy/9_(POS)economy_152.txt', 'Corpus/Input_Data/economy/9_(POS)economy_153.txt', 'Corpus/Input_Data/economy/9_(POS)economy_154.txt', 'Corpus/Input_Data/economy/9_(POS)economy_155.txt', 'Corpus/Input_Data/economy/9_(POS)economy_156.txt', 'Corpus/Input_Data/economy/9_(POS)economy_157.txt', 'Corpus/Input_Data/economy/9_(POS)economy_158.txt', 'Corpus/Input_Data/economy/9_(POS)economy_159.txt', 'Corpus/Input_Data/economy/9_(POS)economy_160.txt', 'Corpus/Input_Data/economy/9_(POS)economy_161.txt', 'Corpus/Input_Data/economy/9_(POS)economy_162.txt', 'Corpus/Input_Data/economy/9_(POS)economy_163.txt', 'Corpus/Input_Data/economy/9_(POS)economy_164.txt', 'Corpus/Input_Data/economy/9_(POS)economy_165.txt', 'Corpus/Input_Data/economy/9_(POS)economy_166.txt'], ['Corpus/Input_Data/education/9_(POS)education_1.txt', 'Corpus/Input_Data/education/9_(POS)education_2.txt', 'Corpus/Input_Data/education/9_(POS)education_3.txt', 'Corpus/Input_Data/education/9_(POS)education_4.txt', 'Corpus/Input_Data/education/9_(POS)education_5.txt', 'Corpus/Input_Data/education/9_(POS)education_6.txt', 'Corpus/Input_Data/education/9_(POS)education_7.txt', 'Corpus/Input_Data/education/9_(POS)education_8.txt', 'Corpus/Input_Data/education/9_(POS)education_9.txt', 'Corpus/Input_Data/education/9_(POS)education_10.txt', 'Corpus/Input_Data/education/9_(POS)education_11.txt', 'Corpus/Input_Data/education/9_(POS)education_12.txt', 'Corpus/Input_Data/education/9_(POS)education_13.txt', 'Corpus/Input_Data/education/9_(POS)education_14.txt', 'Corpus/Input_Data/education/9_(POS)education_15.txt', 'Corpus/Input_Data/education/9_(POS)education_16.txt', 'Corpus/Input_Data/education/9_(POS)education_17.txt', 'Corpus/Input_Data/education/9_(POS)education_18.txt', 'Corpus/Input_Data/education/9_(POS)education_19.txt', 'Corpus/Input_Data/education/9_(POS)education_20.txt', 'Corpus/Input_Data/education/9_(POS)education_21.txt', 'Corpus/Input_Data/education/9_(POS)education_22.txt', 'Corpus/Input_Data/education/9_(POS)education_23.txt', 'Corpus/Input_Data/education/9_(POS)education_24.txt', 'Corpus/Input_Data/education/9_(POS)education_25.txt', 'Corpus/Input_Data/education/9_(POS)education_26.txt', 'Corpus/Input_Data/education/9_(POS)education_27.txt', 'Corpus/Input_Data/education/9_(POS)education_28.txt', 'Corpus/Input_Data/education/9_(POS)education_29.txt', 'Corpus/Input_Data/education/9_(POS)education_30.txt', 'Corpus/Input_Data/education/9_(POS)education_31.txt', 'Corpus/Input_Data/education/9_(POS)education_32.txt', 'Corpus/Input_Data/education/9_(POS)education_33.txt', 'Corpus/Input_Data/education/9_(POS)education_34.txt', 'Corpus/Input_Data/education/9_(POS)education_35.txt', 'Corpus/Input_Data/education/9_(POS)education_36.txt', 'Corpus/Input_Data/education/9_(POS)education_37.txt', 'Corpus/Input_Data/education/9_(POS)education_38.txt', 'Corpus/Input_Data/education/9_(POS)education_39.txt', 'Corpus/Input_Data/education/9_(POS)education_40.txt', 'Corpus/Input_Data/education/9_(POS)education_41.txt', 'Corpus/Input_Data/education/9_(POS)education_42.txt', 'Corpus/Input_Data/education/9_(POS)education_43.txt', 'Corpus/Input_Data/education/9_(POS)education_44.txt', 'Corpus/Input_Data/education/9_(POS)education_45.txt', 'Corpus/Input_Data/education/9_(POS)education_46.txt', 'Corpus/Input_Data/education/9_(POS)education_47.txt', 'Corpus/Input_Data/education/9_(POS)education_48.txt', 'Corpus/Input_Data/education/9_(POS)education_49.txt', 'Corpus/Input_Data/education/9_(POS)education_50.txt', 'Corpus/Input_Data/education/9_(POS)education_51.txt', 'Corpus/Input_Data/education/9_(POS)education_52.txt', 'Corpus/Input_Data/education/9_(POS)education_53.txt', 'Corpus/Input_Data/education/9_(POS)education_54.txt', 'Corpus/Input_Data/education/9_(POS)education_55.txt', 'Corpus/Input_Data/education/9_(POS)education_56.txt', 'Corpus/Input_Data/education/9_(POS)education_57.txt', 'Corpus/Input_Data/education/9_(POS)education_58.txt', 'Corpus/Input_Data/education/9_(POS)education_59.txt', 'Corpus/Input_Data/education/9_(POS)education_60.txt', 'Corpus/Input_Data/education/9_(POS)education_61.txt', 'Corpus/Input_Data/education/9_(POS)education_62.txt', 'Corpus/Input_Data/education/9_(POS)education_63.txt', 'Corpus/Input_Data/education/9_(POS)education_64.txt', 'Corpus/Input_Data/education/9_(POS)education_65.txt', 'Corpus/Input_Data/education/9_(POS)education_66.txt', 'Corpus/Input_Data/education/9_(POS)education_67.txt', 'Corpus/Input_Data/education/9_(POS)education_68.txt', 'Corpus/Input_Data/education/9_(POS)education_69.txt', 'Corpus/Input_Data/education/9_(POS)education_70.txt', 'Corpus/Input_Data/education/9_(POS)education_71.txt', 'Corpus/Input_Data/education/9_(POS)education_72.txt', 'Corpus/Input_Data/education/9_(POS)education_73.txt', 'Corpus/Input_Data/education/9_(POS)education_74.txt', 'Corpus/Input_Data/education/9_(POS)education_75.txt', 'Corpus/Input_Data/education/9_(POS)education_76.txt', 'Corpus/Input_Data/education/9_(POS)education_77.txt', 'Corpus/Input_Data/education/9_(POS)education_78.txt', 'Corpus/Input_Data/education/9_(POS)education_79.txt', 'Corpus/Input_Data/education/9_(POS)education_80.txt', 'Corpus/Input_Data/education/9_(POS)education_81.txt', 'Corpus/Input_Data/education/9_(POS)education_82.txt', 'Corpus/Input_Data/education/9_(POS)education_83.txt', 'Corpus/Input_Data/education/9_(POS)education_84.txt', 'Corpus/Input_Data/education/9_(POS)education_85.txt', 'Corpus/Input_Data/education/9_(POS)education_86.txt', 'Corpus/Input_Data/education/9_(POS)education_87.txt', 'Corpus/Input_Data/education/9_(POS)education_88.txt', 'Corpus/Input_Data/education/9_(POS)education_89.txt', 'Corpus/Input_Data/education/9_(POS)education_90.txt', 'Corpus/Input_Data/education/9_(POS)education_91.txt', 'Corpus/Input_Data/education/9_(POS)education_92.txt', 'Corpus/Input_Data/education/9_(POS)education_93.txt', 'Corpus/Input_Data/education/9_(POS)education_94.txt', 'Corpus/Input_Data/education/9_(POS)education_95.txt', 'Corpus/Input_Data/education/9_(POS)education_96.txt', 'Corpus/Input_Data/education/9_(POS)education_97.txt', 'Corpus/Input_Data/education/9_(POS)education_98.txt', 'Corpus/Input_Data/education/9_(POS)education_99.txt', 'Corpus/Input_Data/education/9_(POS)education_100.txt', 'Corpus/Input_Data/education/9_(POS)education_101.txt', 'Corpus/Input_Data/education/9_(POS)education_102.txt', 'Corpus/Input_Data/education/9_(POS)education_103.txt', 'Corpus/Input_Data/education/9_(POS)education_104.txt', 'Corpus/Input_Data/education/9_(POS)education_105.txt', 'Corpus/Input_Data/education/9_(POS)education_106.txt', 'Corpus/Input_Data/education/9_(POS)education_107.txt', 'Corpus/Input_Data/education/9_(POS)education_108.txt', 'Corpus/Input_Data/education/9_(POS)education_109.txt', 'Corpus/Input_Data/education/9_(POS)education_110.txt', 'Corpus/Input_Data/education/9_(POS)education_111.txt', 'Corpus/Input_Data/education/9_(POS)education_112.txt', 'Corpus/Input_Data/education/9_(POS)education_113.txt', 'Corpus/Input_Data/education/9_(POS)education_114.txt', 'Corpus/Input_Data/education/9_(POS)education_115.txt', 'Corpus/Input_Data/education/9_(POS)education_116.txt', 'Corpus/Input_Data/education/9_(POS)education_117.txt', 'Corpus/Input_Data/education/9_(POS)education_118.txt', 'Corpus/Input_Data/education/9_(POS)education_119.txt', 'Corpus/Input_Data/education/9_(POS)education_120.txt'], ['Corpus/Input_Data/health/9_(POS)health_1.txt', 'Corpus/Input_Data/health/9_(POS)health_2.txt', 'Corpus/Input_Data/health/9_(POS)health_3.txt', 'Corpus/Input_Data/health/9_(POS)health_4.txt', 'Corpus/Input_Data/health/9_(POS)health_5.txt', 'Corpus/Input_Data/health/9_(POS)health_6.txt', 'Corpus/Input_Data/health/9_(POS)health_7.txt', 'Corpus/Input_Data/health/9_(POS)health_8.txt', 'Corpus/Input_Data/health/9_(POS)health_9.txt', 'Corpus/Input_Data/health/9_(POS)health_10.txt', 'Corpus/Input_Data/health/9_(POS)health_11.txt', 'Corpus/Input_Data/health/9_(POS)health_12.txt', 'Corpus/Input_Data/health/9_(POS)health_13.txt', 'Corpus/Input_Data/health/9_(POS)health_14.txt', 'Corpus/Input_Data/health/9_(POS)health_15.txt', 'Corpus/Input_Data/health/9_(POS)health_16.txt', 'Corpus/Input_Data/health/9_(POS)health_17.txt', 'Corpus/Input_Data/health/9_(POS)health_18.txt', 'Corpus/Input_Data/health/9_(POS)health_19.txt', 'Corpus/Input_Data/health/9_(POS)health_20.txt', 'Corpus/Input_Data/health/9_(POS)health_21.txt', 'Corpus/Input_Data/health/9_(POS)health_22.txt', 'Corpus/Input_Data/health/9_(POS)health_23.txt', 'Corpus/Input_Data/health/9_(POS)health_24.txt', 'Corpus/Input_Data/health/9_(POS)health_25.txt', 'Corpus/Input_Data/health/9_(POS)health_26.txt', 'Corpus/Input_Data/health/9_(POS)health_27.txt', 'Corpus/Input_Data/health/9_(POS)health_28.txt', 'Corpus/Input_Data/health/9_(POS)health_29.txt', 'Corpus/Input_Data/health/9_(POS)health_30.txt', 'Corpus/Input_Data/health/9_(POS)health_31.txt', 'Corpus/Input_Data/health/9_(POS)health_32.txt', 'Corpus/Input_Data/health/9_(POS)health_33.txt', 'Corpus/Input_Data/health/9_(POS)health_34.txt', 'Corpus/Input_Data/health/9_(POS)health_35.txt', 'Corpus/Input_Data/health/9_(POS)health_36.txt', 'Corpus/Input_Data/health/9_(POS)health_37.txt', 'Corpus/Input_Data/health/9_(POS)health_38.txt', 'Corpus/Input_Data/health/9_(POS)health_39.txt', 'Corpus/Input_Data/health/9_(POS)health_40.txt', 'Corpus/Input_Data/health/9_(POS)health_41.txt', 'Corpus/Input_Data/health/9_(POS)health_42.txt', 'Corpus/Input_Data/health/9_(POS)health_43.txt', 'Corpus/Input_Data/health/9_(POS)health_44.txt', 'Corpus/Input_Data/health/9_(POS)health_45.txt', 'Corpus/Input_Data/health/9_(POS)health_46.txt', 'Corpus/Input_Data/health/9_(POS)health_47.txt', 'Corpus/Input_Data/health/9_(POS)health_48.txt', 'Corpus/Input_Data/health/9_(POS)health_49.txt', 'Corpus/Input_Data/health/9_(POS)health_50.txt', 'Corpus/Input_Data/health/9_(POS)health_51.txt', 'Corpus/Input_Data/health/9_(POS)health_52.txt', 'Corpus/Input_Data/health/9_(POS)health_53.txt', 'Corpus/Input_Data/health/9_(POS)health_54.txt', 'Corpus/Input_Data/health/9_(POS)health_55.txt', 'Corpus/Input_Data/health/9_(POS)health_56.txt', 'Corpus/Input_Data/health/9_(POS)health_57.txt', 'Corpus/Input_Data/health/9_(POS)health_58.txt', 'Corpus/Input_Data/health/9_(POS)health_59.txt', 'Corpus/Input_Data/health/9_(POS)health_60.txt', 'Corpus/Input_Data/health/9_(POS)health_61.txt', 'Corpus/Input_Data/health/9_(POS)health_62.txt', 'Corpus/Input_Data/health/9_(POS)health_63.txt', 'Corpus/Input_Data/health/9_(POS)health_64.txt', 'Corpus/Input_Data/health/9_(POS)health_65.txt', 'Corpus/Input_Data/health/9_(POS)health_66.txt', 'Corpus/Input_Data/health/9_(POS)health_67.txt', 'Corpus/Input_Data/health/9_(POS)health_68.txt', 'Corpus/Input_Data/health/9_(POS)health_69.txt', 'Corpus/Input_Data/health/9_(POS)health_70.txt', 'Corpus/Input_Data/health/9_(POS)health_71.txt', 'Corpus/Input_Data/health/9_(POS)health_72.txt', 'Corpus/Input_Data/health/9_(POS)health_73.txt', 'Corpus/Input_Data/health/9_(POS)health_74.txt', 'Corpus/Input_Data/health/9_(POS)health_75.txt', 'Corpus/Input_Data/health/9_(POS)health_76.txt', 'Corpus/Input_Data/health/9_(POS)health_77.txt', 'Corpus/Input_Data/health/9_(POS)health_78.txt', 'Corpus/Input_Data/health/9_(POS)health_79.txt', 'Corpus/Input_Data/health/9_(POS)health_80.txt', 'Corpus/Input_Data/health/9_(POS)health_81.txt', 'Corpus/Input_Data/health/9_(POS)health_82.txt', 'Corpus/Input_Data/health/9_(POS)health_83.txt', 'Corpus/Input_Data/health/9_(POS)health_84.txt', 'Corpus/Input_Data/health/9_(POS)health_85.txt', 'Corpus/Input_Data/health/9_(POS)health_86.txt', 'Corpus/Input_Data/health/9_(POS)health_87.txt', 'Corpus/Input_Data/health/9_(POS)health_88.txt', 'Corpus/Input_Data/health/9_(POS)health_89.txt', 'Corpus/Input_Data/health/9_(POS)health_90.txt', 'Corpus/Input_Data/health/9_(POS)health_91.txt', 'Corpus/Input_Data/health/9_(POS)health_92.txt', 'Corpus/Input_Data/health/9_(POS)health_93.txt', 'Corpus/Input_Data/health/9_(POS)health_94.txt', 'Corpus/Input_Data/health/9_(POS)health_95.txt', 'Corpus/Input_Data/health/9_(POS)health_96.txt', 'Corpus/Input_Data/health/9_(POS)health_97.txt', 'Corpus/Input_Data/health/9_(POS)health_98.txt', 'Corpus/Input_Data/health/9_(POS)health_99.txt', 'Corpus/Input_Data/health/9_(POS)health_100.txt', 'Corpus/Input_Data/health/9_(POS)health_101.txt', 'Corpus/Input_Data/health/9_(POS)health_102.txt', 'Corpus/Input_Data/health/9_(POS)health_103.txt', 'Corpus/Input_Data/health/9_(POS)health_104.txt', 'Corpus/Input_Data/health/9_(POS)health_105.txt', 'Corpus/Input_Data/health/9_(POS)health_106.txt', 'Corpus/Input_Data/health/9_(POS)health_107.txt', 'Corpus/Input_Data/health/9_(POS)health_108.txt', 'Corpus/Input_Data/health/9_(POS)health_109.txt', 'Corpus/Input_Data/health/9_(POS)health_110.txt', 'Corpus/Input_Data/health/9_(POS)health_111.txt', 'Corpus/Input_Data/health/9_(POS)health_112.txt', 'Corpus/Input_Data/health/9_(POS)health_113.txt', 'Corpus/Input_Data/health/9_(POS)health_114.txt', 'Corpus/Input_Data/health/9_(POS)health_115.txt', 'Corpus/Input_Data/health/9_(POS)health_116.txt', 'Corpus/Input_Data/health/9_(POS)health_117.txt', 'Corpus/Input_Data/health/9_(POS)health_118.txt', 'Corpus/Input_Data/health/9_(POS)health_119.txt', 'Corpus/Input_Data/health/9_(POS)health_120.txt', 'Corpus/Input_Data/health/9_(POS)health_121.txt', 'Corpus/Input_Data/health/9_(POS)health_122.txt', 'Corpus/Input_Data/health/9_(POS)health_123.txt', 'Corpus/Input_Data/health/9_(POS)health_124.txt', 'Corpus/Input_Data/health/9_(POS)health_125.txt', 'Corpus/Input_Data/health/9_(POS)health_126.txt', 'Corpus/Input_Data/health/9_(POS)health_127.txt', 'Corpus/Input_Data/health/9_(POS)health_128.txt', 'Corpus/Input_Data/health/9_(POS)health_129.txt', 'Corpus/Input_Data/health/9_(POS)health_130.txt', 'Corpus/Input_Data/health/9_(POS)health_131.txt', 'Corpus/Input_Data/health/9_(POS)health_132.txt', 'Corpus/Input_Data/health/9_(POS)health_133.txt', 'Corpus/Input_Data/health/9_(POS)health_134.txt', 'Corpus/Input_Data/health/9_(POS)health_135.txt', 'Corpus/Input_Data/health/9_(POS)health_136.txt', 'Corpus/Input_Data/health/9_(POS)health_137.txt', 'Corpus/Input_Data/health/9_(POS)health_138.txt', 'Corpus/Input_Data/health/9_(POS)health_139.txt', 'Corpus/Input_Data/health/9_(POS)health_140.txt', 'Corpus/Input_Data/health/9_(POS)health_141.txt', 'Corpus/Input_Data/health/9_(POS)health_142.txt', 'Corpus/Input_Data/health/9_(POS)health_143.txt', 'Corpus/Input_Data/health/9_(POS)health_144.txt', 'Corpus/Input_Data/health/9_(POS)health_145.txt', 'Corpus/Input_Data/health/9_(POS)health_146.txt', 'Corpus/Input_Data/health/9_(POS)health_147.txt', 'Corpus/Input_Data/health/9_(POS)health_148.txt', 'Corpus/Input_Data/health/9_(POS)health_149.txt', 'Corpus/Input_Data/health/9_(POS)health_150.txt', 'Corpus/Input_Data/health/9_(POS)health_151.txt', 'Corpus/Input_Data/health/9_(POS)health_152.txt', 'Corpus/Input_Data/health/9_(POS)health_153.txt', 'Corpus/Input_Data/health/9_(POS)health_154.txt', 'Corpus/Input_Data/health/9_(POS)health_155.txt', 'Corpus/Input_Data/health/9_(POS)health_156.txt', 'Corpus/Input_Data/health/9_(POS)health_157.txt', 'Corpus/Input_Data/health/9_(POS)health_158.txt', 'Corpus/Input_Data/health/9_(POS)health_159.txt', 'Corpus/Input_Data/health/9_(POS)health_160.txt', 'Corpus/Input_Data/health/9_(POS)health_161.txt', 'Corpus/Input_Data/health/9_(POS)health_162.txt', 'Corpus/Input_Data/health/9_(POS)health_163.txt', 'Corpus/Input_Data/health/9_(POS)health_164.txt', 'Corpus/Input_Data/health/9_(POS)health_165.txt', 'Corpus/Input_Data/health/9_(POS)health_166.txt', 'Corpus/Input_Data/health/9_(POS)health_167.txt', 'Corpus/Input_Data/health/9_(POS)health_168.txt', 'Corpus/Input_Data/health/9_(POS)health_169.txt', 'Corpus/Input_Data/health/9_(POS)health_170.txt', 'Corpus/Input_Data/health/9_(POS)health_171.txt', 'Corpus/Input_Data/health/9_(POS)health_172.txt', 'Corpus/Input_Data/health/9_(POS)health_173.txt', 'Corpus/Input_Data/health/9_(POS)health_174.txt', 'Corpus/Input_Data/health/9_(POS)health_175.txt', 'Corpus/Input_Data/health/9_(POS)health_176.txt', 'Corpus/Input_Data/health/9_(POS)health_177.txt', 'Corpus/Input_Data/health/9_(POS)health_178.txt', 'Corpus/Input_Data/health/9_(POS)health_179.txt', 'Corpus/Input_Data/health/9_(POS)health_180.txt', 'Corpus/Input_Data/health/9_(POS)health_181.txt', 'Corpus/Input_Data/health/9_(POS)health_182.txt', 'Corpus/Input_Data/health/9_(POS)health_183.txt', 'Corpus/Input_Data/health/9_(POS)health_184.txt', 'Corpus/Input_Data/health/9_(POS)health_185.txt', 'Corpus/Input_Data/health/9_(POS)health_186.txt', 'Corpus/Input_Data/health/9_(POS)health_187.txt', 'Corpus/Input_Data/health/9_(POS)health_188.txt', 'Corpus/Input_Data/health/9_(POS)health_189.txt', 'Corpus/Input_Data/health/9_(POS)health_190.txt', 'Corpus/Input_Data/health/9_(POS)health_191.txt'], ['Corpus/Input_Data/life/9_(POS)life_1.txt', 'Corpus/Input_Data/life/9_(POS)life_2.txt', 'Corpus/Input_Data/life/9_(POS)life_3.txt', 'Corpus/Input_Data/life/9_(POS)life_4.txt', 'Corpus/Input_Data/life/9_(POS)life_5.txt', 'Corpus/Input_Data/life/9_(POS)life_6.txt', 'Corpus/Input_Data/life/9_(POS)life_7.txt', 'Corpus/Input_Data/life/9_(POS)life_8.txt', 'Corpus/Input_Data/life/9_(POS)life_9.txt', 'Corpus/Input_Data/life/9_(POS)life_10.txt', 'Corpus/Input_Data/life/9_(POS)life_11.txt', 'Corpus/Input_Data/life/9_(POS)life_12.txt', 'Corpus/Input_Data/life/9_(POS)life_13.txt', 'Corpus/Input_Data/life/9_(POS)life_14.txt', 'Corpus/Input_Data/life/9_(POS)life_15.txt', 'Corpus/Input_Data/life/9_(POS)life_16.txt', 'Corpus/Input_Data/life/9_(POS)life_17.txt', 'Corpus/Input_Data/life/9_(POS)life_18.txt', 'Corpus/Input_Data/life/9_(POS)life_19.txt', 'Corpus/Input_Data/life/9_(POS)life_20.txt', 'Corpus/Input_Data/life/9_(POS)life_21.txt', 'Corpus/Input_Data/life/9_(POS)life_22.txt', 'Corpus/Input_Data/life/9_(POS)life_23.txt', 'Corpus/Input_Data/life/9_(POS)life_24.txt', 'Corpus/Input_Data/life/9_(POS)life_25.txt', 'Corpus/Input_Data/life/9_(POS)life_26.txt', 'Corpus/Input_Data/life/9_(POS)life_27.txt', 'Corpus/Input_Data/life/9_(POS)life_28.txt', 'Corpus/Input_Data/life/9_(POS)life_29.txt', 'Corpus/Input_Data/life/9_(POS)life_30.txt', 'Corpus/Input_Data/life/9_(POS)life_31.txt', 'Corpus/Input_Data/life/9_(POS)life_32.txt', 'Corpus/Input_Data/life/9_(POS)life_33.txt', 'Corpus/Input_Data/life/9_(POS)life_34.txt', 'Corpus/Input_Data/life/9_(POS)life_35.txt', 'Corpus/Input_Data/life/9_(POS)life_36.txt', 'Corpus/Input_Data/life/9_(POS)life_37.txt', 'Corpus/Input_Data/life/9_(POS)life_38.txt', 'Corpus/Input_Data/life/9_(POS)life_39.txt', 'Corpus/Input_Data/life/9_(POS)life_40.txt', 'Corpus/Input_Data/life/9_(POS)life_41.txt', 'Corpus/Input_Data/life/9_(POS)life_42.txt', 'Corpus/Input_Data/life/9_(POS)life_43.txt', 'Corpus/Input_Data/life/9_(POS)life_44.txt', 'Corpus/Input_Data/life/9_(POS)life_45.txt', 'Corpus/Input_Data/life/9_(POS)life_46.txt', 'Corpus/Input_Data/life/9_(POS)life_47.txt', 'Corpus/Input_Data/life/9_(POS)life_48.txt', 'Corpus/Input_Data/life/9_(POS)life_49.txt', 'Corpus/Input_Data/life/9_(POS)life_50.txt', 'Corpus/Input_Data/life/9_(POS)life_51.txt', 'Corpus/Input_Data/life/9_(POS)life_52.txt', 'Corpus/Input_Data/life/9_(POS)life_53.txt', 'Corpus/Input_Data/life/9_(POS)life_54.txt', 'Corpus/Input_Data/life/9_(POS)life_55.txt', 'Corpus/Input_Data/life/9_(POS)life_56.txt', 'Corpus/Input_Data/life/9_(POS)life_57.txt', 'Corpus/Input_Data/life/9_(POS)life_58.txt', 'Corpus/Input_Data/life/9_(POS)life_59.txt', 'Corpus/Input_Data/life/9_(POS)life_60.txt', 'Corpus/Input_Data/life/9_(POS)life_61.txt', 'Corpus/Input_Data/life/9_(POS)life_62.txt', 'Corpus/Input_Data/life/9_(POS)life_63.txt', 'Corpus/Input_Data/life/9_(POS)life_64.txt', 'Corpus/Input_Data/life/9_(POS)life_65.txt', 'Corpus/Input_Data/life/9_(POS)life_66.txt', 'Corpus/Input_Data/life/9_(POS)life_67.txt', 'Corpus/Input_Data/life/9_(POS)life_68.txt', 'Corpus/Input_Data/life/9_(POS)life_69.txt', 'Corpus/Input_Data/life/9_(POS)life_70.txt', 'Corpus/Input_Data/life/9_(POS)life_71.txt', 'Corpus/Input_Data/life/9_(POS)life_72.txt', 'Corpus/Input_Data/life/9_(POS)life_73.txt', 'Corpus/Input_Data/life/9_(POS)life_74.txt', 'Corpus/Input_Data/life/9_(POS)life_75.txt', 'Corpus/Input_Data/life/9_(POS)life_76.txt', 'Corpus/Input_Data/life/9_(POS)life_77.txt', 'Corpus/Input_Data/life/9_(POS)life_78.txt', 'Corpus/Input_Data/life/9_(POS)life_79.txt', 'Corpus/Input_Data/life/9_(POS)life_80.txt', 'Corpus/Input_Data/life/9_(POS)life_81.txt', 'Corpus/Input_Data/life/9_(POS)life_82.txt', 'Corpus/Input_Data/life/9_(POS)life_83.txt', 'Corpus/Input_Data/life/9_(POS)life_84.txt', 'Corpus/Input_Data/life/9_(POS)life_85.txt', 'Corpus/Input_Data/life/9_(POS)life_86.txt', 'Corpus/Input_Data/life/9_(POS)life_87.txt', 'Corpus/Input_Data/life/9_(POS)life_88.txt', 'Corpus/Input_Data/life/9_(POS)life_89.txt', 'Corpus/Input_Data/life/9_(POS)life_90.txt', 'Corpus/Input_Data/life/9_(POS)life_91.txt', 'Corpus/Input_Data/life/9_(POS)life_92.txt', 'Corpus/Input_Data/life/9_(POS)life_93.txt', 'Corpus/Input_Data/life/9_(POS)life_94.txt', 'Corpus/Input_Data/life/9_(POS)life_95.txt', 'Corpus/Input_Data/life/9_(POS)life_96.txt', 'Corpus/Input_Data/life/9_(POS)life_97.txt', 'Corpus/Input_Data/life/9_(POS)life_98.txt', 'Corpus/Input_Data/life/9_(POS)life_99.txt', 'Corpus/Input_Data/life/9_(POS)life_100.txt', 'Corpus/Input_Data/life/9_(POS)life_101.txt', 'Corpus/Input_Data/life/9_(POS)life_102.txt', 'Corpus/Input_Data/life/9_(POS)life_103.txt', 'Corpus/Input_Data/life/9_(POS)life_104.txt', 'Corpus/Input_Data/life/9_(POS)life_105.txt', 'Corpus/Input_Data/life/9_(POS)life_106.txt', 'Corpus/Input_Data/life/9_(POS)life_107.txt', 'Corpus/Input_Data/life/9_(POS)life_108.txt', 'Corpus/Input_Data/life/9_(POS)life_109.txt', 'Corpus/Input_Data/life/9_(POS)life_110.txt', 'Corpus/Input_Data/life/9_(POS)life_111.txt', 'Corpus/Input_Data/life/9_(POS)life_112.txt'], ['Corpus/Input_Data/person/9_(POS)person_1.txt', 'Corpus/Input_Data/person/9_(POS)person_2.txt', 'Corpus/Input_Data/person/9_(POS)person_3.txt', 'Corpus/Input_Data/person/9_(POS)person_4.txt', 'Corpus/Input_Data/person/9_(POS)person_5.txt', 'Corpus/Input_Data/person/9_(POS)person_6.txt', 'Corpus/Input_Data/person/9_(POS)person_7.txt', 'Corpus/Input_Data/person/9_(POS)person_8.txt', 'Corpus/Input_Data/person/9_(POS)person_9.txt', 'Corpus/Input_Data/person/9_(POS)person_10.txt', 'Corpus/Input_Data/person/9_(POS)person_11.txt', 'Corpus/Input_Data/person/9_(POS)person_12.txt', 'Corpus/Input_Data/person/9_(POS)person_13.txt', 'Corpus/Input_Data/person/9_(POS)person_14.txt', 'Corpus/Input_Data/person/9_(POS)person_15.txt', 'Corpus/Input_Data/person/9_(POS)person_16.txt', 'Corpus/Input_Data/person/9_(POS)person_17.txt', 'Corpus/Input_Data/person/9_(POS)person_18.txt', 'Corpus/Input_Data/person/9_(POS)person_19.txt', 'Corpus/Input_Data/person/9_(POS)person_20.txt', 'Corpus/Input_Data/person/9_(POS)person_21.txt', 'Corpus/Input_Data/person/9_(POS)person_22.txt', 'Corpus/Input_Data/person/9_(POS)person_23.txt', 'Corpus/Input_Data/person/9_(POS)person_24.txt', 'Corpus/Input_Data/person/9_(POS)person_25.txt', 'Corpus/Input_Data/person/9_(POS)person_26.txt', 'Corpus/Input_Data/person/9_(POS)person_27.txt', 'Corpus/Input_Data/person/9_(POS)person_28.txt', 'Corpus/Input_Data/person/9_(POS)person_29.txt', 'Corpus/Input_Data/person/9_(POS)person_30.txt', 'Corpus/Input_Data/person/9_(POS)person_31.txt', 'Corpus/Input_Data/person/9_(POS)person_32.txt', 'Corpus/Input_Data/person/9_(POS)person_33.txt', 'Corpus/Input_Data/person/9_(POS)person_34.txt', 'Corpus/Input_Data/person/9_(POS)person_35.txt', 'Corpus/Input_Data/person/9_(POS)person_36.txt', 'Corpus/Input_Data/person/9_(POS)person_37.txt', 'Corpus/Input_Data/person/9_(POS)person_38.txt', 'Corpus/Input_Data/person/9_(POS)person_39.txt', 'Corpus/Input_Data/person/9_(POS)person_40.txt', 'Corpus/Input_Data/person/9_(POS)person_41.txt', 'Corpus/Input_Data/person/9_(POS)person_42.txt', 'Corpus/Input_Data/person/9_(POS)person_43.txt', 'Corpus/Input_Data/person/9_(POS)person_44.txt', 'Corpus/Input_Data/person/9_(POS)person_45.txt', 'Corpus/Input_Data/person/9_(POS)person_46.txt', 'Corpus/Input_Data/person/9_(POS)person_47.txt', 'Corpus/Input_Data/person/9_(POS)person_48.txt', 'Corpus/Input_Data/person/9_(POS)person_49.txt', 'Corpus/Input_Data/person/9_(POS)person_50.txt', 'Corpus/Input_Data/person/9_(POS)person_51.txt', 'Corpus/Input_Data/person/9_(POS)person_52.txt', 'Corpus/Input_Data/person/9_(POS)person_53.txt', 'Corpus/Input_Data/person/9_(POS)person_54.txt', 'Corpus/Input_Data/person/9_(POS)person_55.txt', 'Corpus/Input_Data/person/9_(POS)person_56.txt', 'Corpus/Input_Data/person/9_(POS)person_57.txt', 'Corpus/Input_Data/person/9_(POS)person_58.txt', 'Corpus/Input_Data/person/9_(POS)person_59.txt', 'Corpus/Input_Data/person/9_(POS)person_60.txt', 'Corpus/Input_Data/person/9_(POS)person_61.txt', 'Corpus/Input_Data/person/9_(POS)person_62.txt', 'Corpus/Input_Data/person/9_(POS)person_63.txt', 'Corpus/Input_Data/person/9_(POS)person_64.txt', 'Corpus/Input_Data/person/9_(POS)person_65.txt', 'Corpus/Input_Data/person/9_(POS)person_66.txt', 'Corpus/Input_Data/person/9_(POS)person_67.txt', 'Corpus/Input_Data/person/9_(POS)person_68.txt', 'Corpus/Input_Data/person/9_(POS)person_69.txt', 'Corpus/Input_Data/person/9_(POS)person_70.txt', 'Corpus/Input_Data/person/9_(POS)person_71.txt', 'Corpus/Input_Data/person/9_(POS)person_72.txt', 'Corpus/Input_Data/person/9_(POS)person_73.txt', 'Corpus/Input_Data/person/9_(POS)person_74.txt', 'Corpus/Input_Data/person/9_(POS)person_75.txt', 'Corpus/Input_Data/person/9_(POS)person_76.txt', 'Corpus/Input_Data/person/9_(POS)person_77.txt', 'Corpus/Input_Data/person/9_(POS)person_78.txt', 'Corpus/Input_Data/person/9_(POS)person_79.txt', 'Corpus/Input_Data/person/9_(POS)person_80.txt', 'Corpus/Input_Data/person/9_(POS)person_81.txt', 'Corpus/Input_Data/person/9_(POS)person_82.txt', 'Corpus/Input_Data/person/9_(POS)person_83.txt', 'Corpus/Input_Data/person/9_(POS)person_84.txt', 'Corpus/Input_Data/person/9_(POS)person_85.txt', 'Corpus/Input_Data/person/9_(POS)person_86.txt', 'Corpus/Input_Data/person/9_(POS)person_87.txt', 'Corpus/Input_Data/person/9_(POS)person_88.txt', 'Corpus/Input_Data/person/9_(POS)person_89.txt', 'Corpus/Input_Data/person/9_(POS)person_90.txt', 'Corpus/Input_Data/person/9_(POS)person_91.txt', 'Corpus/Input_Data/person/9_(POS)person_92.txt', 'Corpus/Input_Data/person/9_(POS)person_93.txt', 'Corpus/Input_Data/person/9_(POS)person_94.txt', 'Corpus/Input_Data/person/9_(POS)person_95.txt', 'Corpus/Input_Data/person/9_(POS)person_96.txt', 'Corpus/Input_Data/person/9_(POS)person_97.txt', 'Corpus/Input_Data/person/9_(POS)person_98.txt', 'Corpus/Input_Data/person/9_(POS)person_99.txt', 'Corpus/Input_Data/person/9_(POS)person_100.txt', 'Corpus/Input_Data/person/9_(POS)person_101.txt', 'Corpus/Input_Data/person/9_(POS)person_102.txt', 'Corpus/Input_Data/person/9_(POS)person_103.txt', 'Corpus/Input_Data/person/9_(POS)person_104.txt', 'Corpus/Input_Data/person/9_(POS)person_105.txt', 'Corpus/Input_Data/person/9_(POS)person_106.txt', 'Corpus/Input_Data/person/9_(POS)person_107.txt', 'Corpus/Input_Data/person/9_(POS)person_108.txt', 'Corpus/Input_Data/person/9_(POS)person_109.txt', 'Corpus/Input_Data/person/9_(POS)person_110.txt', 'Corpus/Input_Data/person/9_(POS)person_111.txt', 'Corpus/Input_Data/person/9_(POS)person_112.txt', 'Corpus/Input_Data/person/9_(POS)person_113.txt', 'Corpus/Input_Data/person/9_(POS)person_114.txt', 'Corpus/Input_Data/person/9_(POS)person_115.txt', 'Corpus/Input_Data/person/9_(POS)person_116.txt', 'Corpus/Input_Data/person/9_(POS)person_117.txt', 'Corpus/Input_Data/person/9_(POS)person_118.txt', 'Corpus/Input_Data/person/9_(POS)person_119.txt', 'Corpus/Input_Data/person/9_(POS)person_120.txt', 'Corpus/Input_Data/person/9_(POS)person_121.txt', 'Corpus/Input_Data/person/9_(POS)person_122.txt', 'Corpus/Input_Data/person/9_(POS)person_123.txt', 'Corpus/Input_Data/person/9_(POS)person_124.txt', 'Corpus/Input_Data/person/9_(POS)person_125.txt', 'Corpus/Input_Data/person/9_(POS)person_126.txt', 'Corpus/Input_Data/person/9_(POS)person_127.txt', 'Corpus/Input_Data/person/9_(POS)person_128.txt', 'Corpus/Input_Data/person/9_(POS)person_129.txt', 'Corpus/Input_Data/person/9_(POS)person_130.txt', 'Corpus/Input_Data/person/9_(POS)person_131.txt', 'Corpus/Input_Data/person/9_(POS)person_132.txt', 'Corpus/Input_Data/person/9_(POS)person_133.txt', 'Corpus/Input_Data/person/9_(POS)person_134.txt', 'Corpus/Input_Data/person/9_(POS)person_135.txt', 'Corpus/Input_Data/person/9_(POS)person_136.txt', 'Corpus/Input_Data/person/9_(POS)person_137.txt', 'Corpus/Input_Data/person/9_(POS)person_138.txt', 'Corpus/Input_Data/person/9_(POS)person_139.txt', 'Corpus/Input_Data/person/9_(POS)person_140.txt', 'Corpus/Input_Data/person/9_(POS)person_141.txt', 'Corpus/Input_Data/person/9_(POS)person_142.txt', 'Corpus/Input_Data/person/9_(POS)person_143.txt', 'Corpus/Input_Data/person/9_(POS)person_144.txt', 'Corpus/Input_Data/person/9_(POS)person_145.txt', 'Corpus/Input_Data/person/9_(POS)person_146.txt', 'Corpus/Input_Data/person/9_(POS)person_147.txt', 'Corpus/Input_Data/person/9_(POS)person_148.txt', 'Corpus/Input_Data/person/9_(POS)person_149.txt', 'Corpus/Input_Data/person/9_(POS)person_150.txt', 'Corpus/Input_Data/person/9_(POS)person_151.txt', 'Corpus/Input_Data/person/9_(POS)person_152.txt', 'Corpus/Input_Data/person/9_(POS)person_153.txt', 'Corpus/Input_Data/person/9_(POS)person_154.txt', 'Corpus/Input_Data/person/9_(POS)person_155.txt', 'Corpus/Input_Data/person/9_(POS)person_156.txt', 'Corpus/Input_Data/person/9_(POS)person_157.txt', 'Corpus/Input_Data/person/9_(POS)person_158.txt', 'Corpus/Input_Data/person/9_(POS)person_159.txt', 'Corpus/Input_Data/person/9_(POS)person_160.txt', 'Corpus/Input_Data/person/9_(POS)person_161.txt', 'Corpus/Input_Data/person/9_(POS)person_162.txt', 'Corpus/Input_Data/person/9_(POS)person_163.txt', 'Corpus/Input_Data/person/9_(POS)person_164.txt', 'Corpus/Input_Data/person/9_(POS)person_165.txt', 'Corpus/Input_Data/person/9_(POS)person_166.txt', 'Corpus/Input_Data/person/9_(POS)person_167.txt', 'Corpus/Input_Data/person/9_(POS)person_168.txt', 'Corpus/Input_Data/person/9_(POS)person_169.txt', 'Corpus/Input_Data/person/9_(POS)person_170.txt', 'Corpus/Input_Data/person/9_(POS)person_171.txt', 'Corpus/Input_Data/person/9_(POS)person_172.txt'], ['Corpus/Input_Data/policy/9_(POS)policy_1.txt', 'Corpus/Input_Data/policy/9_(POS)policy_2.txt', 'Corpus/Input_Data/policy/9_(POS)policy_3.txt', 'Corpus/Input_Data/policy/9_(POS)policy_4.txt', 'Corpus/Input_Data/policy/9_(POS)policy_5.txt', 'Corpus/Input_Data/policy/9_(POS)policy_6.txt', 'Corpus/Input_Data/policy/9_(POS)policy_7.txt', 'Corpus/Input_Data/policy/9_(POS)policy_8.txt', 'Corpus/Input_Data/policy/9_(POS)policy_9.txt', 'Corpus/Input_Data/policy/9_(POS)policy_10.txt', 'Corpus/Input_Data/policy/9_(POS)policy_11.txt', 'Corpus/Input_Data/policy/9_(POS)policy_12.txt', 'Corpus/Input_Data/policy/9_(POS)policy_13.txt', 'Corpus/Input_Data/policy/9_(POS)policy_14.txt', 'Corpus/Input_Data/policy/9_(POS)policy_15.txt', 'Corpus/Input_Data/policy/9_(POS)policy_16.txt', 'Corpus/Input_Data/policy/9_(POS)policy_17.txt', 'Corpus/Input_Data/policy/9_(POS)policy_18.txt', 'Corpus/Input_Data/policy/9_(POS)policy_19.txt', 'Corpus/Input_Data/policy/9_(POS)policy_20.txt', 'Corpus/Input_Data/policy/9_(POS)policy_21.txt', 'Corpus/Input_Data/policy/9_(POS)policy_22.txt', 'Corpus/Input_Data/policy/9_(POS)policy_23.txt', 'Corpus/Input_Data/policy/9_(POS)policy_24.txt', 'Corpus/Input_Data/policy/9_(POS)policy_25.txt', 'Corpus/Input_Data/policy/9_(POS)policy_26.txt', 'Corpus/Input_Data/policy/9_(POS)policy_27.txt', 'Corpus/Input_Data/policy/9_(POS)policy_28.txt', 'Corpus/Input_Data/policy/9_(POS)policy_29.txt', 'Corpus/Input_Data/policy/9_(POS)policy_30.txt', 'Corpus/Input_Data/policy/9_(POS)policy_31.txt', 'Corpus/Input_Data/policy/9_(POS)policy_32.txt', 'Corpus/Input_Data/policy/9_(POS)policy_33.txt', 'Corpus/Input_Data/policy/9_(POS)policy_34.txt', 'Corpus/Input_Data/policy/9_(POS)policy_35.txt', 'Corpus/Input_Data/policy/9_(POS)policy_36.txt', 'Corpus/Input_Data/policy/9_(POS)policy_37.txt', 'Corpus/Input_Data/policy/9_(POS)policy_38.txt', 'Corpus/Input_Data/policy/9_(POS)policy_39.txt', 'Corpus/Input_Data/policy/9_(POS)policy_40.txt', 'Corpus/Input_Data/policy/9_(POS)policy_41.txt', 'Corpus/Input_Data/policy/9_(POS)policy_42.txt', 'Corpus/Input_Data/policy/9_(POS)policy_43.txt', 'Corpus/Input_Data/policy/9_(POS)policy_44.txt', 'Corpus/Input_Data/policy/9_(POS)policy_45.txt', 'Corpus/Input_Data/policy/9_(POS)policy_46.txt', 'Corpus/Input_Data/policy/9_(POS)policy_47.txt', 'Corpus/Input_Data/policy/9_(POS)policy_48.txt', 'Corpus/Input_Data/policy/9_(POS)policy_49.txt', 'Corpus/Input_Data/policy/9_(POS)policy_50.txt', 'Corpus/Input_Data/policy/9_(POS)policy_51.txt', 'Corpus/Input_Data/policy/9_(POS)policy_52.txt', 'Corpus/Input_Data/policy/9_(POS)policy_53.txt', 'Corpus/Input_Data/policy/9_(POS)policy_54.txt', 'Corpus/Input_Data/policy/9_(POS)policy_55.txt', 'Corpus/Input_Data/policy/9_(POS)policy_56.txt', 'Corpus/Input_Data/policy/9_(POS)policy_57.txt', 'Corpus/Input_Data/policy/9_(POS)policy_58.txt', 'Corpus/Input_Data/policy/9_(POS)policy_59.txt', 'Corpus/Input_Data/policy/9_(POS)policy_60.txt', 'Corpus/Input_Data/policy/9_(POS)policy_61.txt', 'Corpus/Input_Data/policy/9_(POS)policy_62.txt', 'Corpus/Input_Data/policy/9_(POS)policy_63.txt', 'Corpus/Input_Data/policy/9_(POS)policy_64.txt', 'Corpus/Input_Data/policy/9_(POS)policy_65.txt', 'Corpus/Input_Data/policy/9_(POS)policy_66.txt', 'Corpus/Input_Data/policy/9_(POS)policy_67.txt', 'Corpus/Input_Data/policy/9_(POS)policy_68.txt', 'Corpus/Input_Data/policy/9_(POS)policy_69.txt', 'Corpus/Input_Data/policy/9_(POS)policy_70.txt', 'Corpus/Input_Data/policy/9_(POS)policy_71.txt', 'Corpus/Input_Data/policy/9_(POS)policy_72.txt', 'Corpus/Input_Data/policy/9_(POS)policy_73.txt', 'Corpus/Input_Data/policy/9_(POS)policy_74.txt', 'Corpus/Input_Data/policy/9_(POS)policy_75.txt', 'Corpus/Input_Data/policy/9_(POS)policy_76.txt', 'Corpus/Input_Data/policy/9_(POS)policy_77.txt', 'Corpus/Input_Data/policy/9_(POS)policy_78.txt', 'Corpus/Input_Data/policy/9_(POS)policy_79.txt', 'Corpus/Input_Data/policy/9_(POS)policy_80.txt', 'Corpus/Input_Data/policy/9_(POS)policy_81.txt', 'Corpus/Input_Data/policy/9_(POS)policy_82.txt', 'Corpus/Input_Data/policy/9_(POS)policy_83.txt', 'Corpus/Input_Data/policy/9_(POS)policy_84.txt', 'Corpus/Input_Data/policy/9_(POS)policy_85.txt', 'Corpus/Input_Data/policy/9_(POS)policy_86.txt', 'Corpus/Input_Data/policy/9_(POS)policy_87.txt', 'Corpus/Input_Data/policy/9_(POS)policy_88.txt', 'Corpus/Input_Data/policy/9_(POS)policy_89.txt', 'Corpus/Input_Data/policy/9_(POS)policy_90.txt', 'Corpus/Input_Data/policy/9_(POS)policy_91.txt', 'Corpus/Input_Data/policy/9_(POS)policy_92.txt', 'Corpus/Input_Data/policy/9_(POS)policy_93.txt', 'Corpus/Input_Data/policy/9_(POS)policy_94.txt', 'Corpus/Input_Data/policy/9_(POS)policy_95.txt', 'Corpus/Input_Data/policy/9_(POS)policy_96.txt', 'Corpus/Input_Data/policy/9_(POS)policy_97.txt', 'Corpus/Input_Data/policy/9_(POS)policy_98.txt', 'Corpus/Input_Data/policy/9_(POS)policy_99.txt', 'Corpus/Input_Data/policy/9_(POS)policy_100.txt', 'Corpus/Input_Data/policy/9_(POS)policy_101.txt', 'Corpus/Input_Data/policy/9_(POS)policy_102.txt', 'Corpus/Input_Data/policy/9_(POS)policy_103.txt', 'Corpus/Input_Data/policy/9_(POS)policy_104.txt', 'Corpus/Input_Data/policy/9_(POS)policy_105.txt', 'Corpus/Input_Data/policy/9_(POS)policy_106.txt', 'Corpus/Input_Data/policy/9_(POS)policy_107.txt', 'Corpus/Input_Data/policy/9_(POS)policy_108.txt', 'Corpus/Input_Data/policy/9_(POS)policy_109.txt', 'Corpus/Input_Data/policy/9_(POS)policy_110.txt', 'Corpus/Input_Data/policy/9_(POS)policy_111.txt', 'Corpus/Input_Data/policy/9_(POS)policy_112.txt', 'Corpus/Input_Data/policy/9_(POS)policy_113.txt', 'Corpus/Input_Data/policy/9_(POS)policy_114.txt', 'Corpus/Input_Data/policy/9_(POS)policy_115.txt', 'Corpus/Input_Data/policy/9_(POS)policy_116.txt', 'Corpus/Input_Data/policy/9_(POS)policy_117.txt', 'Corpus/Input_Data/policy/9_(POS)policy_118.txt', 'Corpus/Input_Data/policy/9_(POS)policy_119.txt', 'Corpus/Input_Data/policy/9_(POS)policy_120.txt', 'Corpus/Input_Data/policy/9_(POS)policy_121.txt', 'Corpus/Input_Data/policy/9_(POS)policy_122.txt', 'Corpus/Input_Data/policy/9_(POS)policy_123.txt', 'Corpus/Input_Data/policy/9_(POS)policy_124.txt', 'Corpus/Input_Data/policy/9_(POS)policy_125.txt', 'Corpus/Input_Data/policy/9_(POS)policy_126.txt', 'Corpus/Input_Data/policy/9_(POS)policy_127.txt', 'Corpus/Input_Data/policy/9_(POS)policy_128.txt', 'Corpus/Input_Data/policy/9_(POS)policy_129.txt', 'Corpus/Input_Data/policy/9_(POS)policy_130.txt', 'Corpus/Input_Data/policy/9_(POS)policy_131.txt', 'Corpus/Input_Data/policy/9_(POS)policy_132.txt', 'Corpus/Input_Data/policy/9_(POS)policy_133.txt', 'Corpus/Input_Data/policy/9_(POS)policy_134.txt', 'Corpus/Input_Data/policy/9_(POS)policy_135.txt', 'Corpus/Input_Data/policy/9_(POS)policy_136.txt', 'Corpus/Input_Data/policy/9_(POS)policy_137.txt', 'Corpus/Input_Data/policy/9_(POS)policy_138.txt', 'Corpus/Input_Data/policy/9_(POS)policy_139.txt', 'Corpus/Input_Data/policy/9_(POS)policy_140.txt', 'Corpus/Input_Data/policy/9_(POS)policy_141.txt', 'Corpus/Input_Data/policy/9_(POS)policy_142.txt', 'Corpus/Input_Data/policy/9_(POS)policy_143.txt', 'Corpus/Input_Data/policy/9_(POS)policy_144.txt', 'Corpus/Input_Data/policy/9_(POS)policy_145.txt', 'Corpus/Input_Data/policy/9_(POS)policy_146.txt', 'Corpus/Input_Data/policy/9_(POS)policy_147.txt', 'Corpus/Input_Data/policy/9_(POS)policy_148.txt', 'Corpus/Input_Data/policy/9_(POS)policy_149.txt', 'Corpus/Input_Data/policy/9_(POS)policy_150.txt', 'Corpus/Input_Data/policy/9_(POS)policy_151.txt', 'Corpus/Input_Data/policy/9_(POS)policy_152.txt', 'Corpus/Input_Data/policy/9_(POS)policy_153.txt', 'Corpus/Input_Data/policy/9_(POS)policy_154.txt', 'Corpus/Input_Data/policy/9_(POS)policy_155.txt', 'Corpus/Input_Data/policy/9_(POS)policy_156.txt', 'Corpus/Input_Data/policy/9_(POS)policy_157.txt', 'Corpus/Input_Data/policy/9_(POS)policy_158.txt', 'Corpus/Input_Data/policy/9_(POS)policy_159.txt', 'Corpus/Input_Data/policy/9_(POS)policy_160.txt', 'Corpus/Input_Data/policy/9_(POS)policy_161.txt', 'Corpus/Input_Data/policy/9_(POS)policy_162.txt', 'Corpus/Input_Data/policy/9_(POS)policy_163.txt', 'Corpus/Input_Data/policy/9_(POS)policy_164.txt', 'Corpus/Input_Data/policy/9_(POS)policy_165.txt', 'Corpus/Input_Data/policy/9_(POS)policy_166.txt', 'Corpus/Input_Data/policy/9_(POS)policy_167.txt', 'Corpus/Input_Data/policy/9_(POS)policy_168.txt', 'Corpus/Input_Data/policy/9_(POS)policy_169.txt', 'Corpus/Input_Data/policy/9_(POS)policy_170.txt', 'Corpus/Input_Data/policy/9_(POS)policy_171.txt', 'Corpus/Input_Data/policy/9_(POS)policy_172.txt', 'Corpus/Input_Data/policy/9_(POS)policy_173.txt', 'Corpus/Input_Data/policy/9_(POS)policy_174.txt', 'Corpus/Input_Data/policy/9_(POS)policy_175.txt', 'Corpus/Input_Data/policy/9_(POS)policy_176.txt', 'Corpus/Input_Data/policy/9_(POS)policy_177.txt', 'Corpus/Input_Data/policy/9_(POS)policy_178.txt', 'Corpus/Input_Data/policy/9_(POS)policy_179.txt', 'Corpus/Input_Data/policy/9_(POS)policy_180.txt', 'Corpus/Input_Data/policy/9_(POS)policy_181.txt', 'Corpus/Input_Data/policy/9_(POS)policy_182.txt', 'Corpus/Input_Data/policy/9_(POS)policy_183.txt', 'Corpus/Input_Data/policy/9_(POS)policy_184.txt', 'Corpus/Input_Data/policy/9_(POS)policy_185.txt', 'Corpus/Input_Data/policy/9_(POS)policy_186.txt', 'Corpus/Input_Data/policy/9_(POS)policy_187.txt', 'Corpus/Input_Data/policy/9_(POS)policy_188.txt', 'Corpus/Input_Data/policy/9_(POS)policy_189.txt', 'Corpus/Input_Data/policy/9_(POS)policy_190.txt', 'Corpus/Input_Data/policy/9_(POS)policy_191.txt', 'Corpus/Input_Data/policy/9_(POS)policy_192.txt', 'Corpus/Input_Data/policy/9_(POS)policy_193.txt', 'Corpus/Input_Data/policy/9_(POS)policy_194.txt', 'Corpus/Input_Data/policy/9_(POS)policy_195.txt', 'Corpus/Input_Data/policy/9_(POS)policy_196.txt', 'Corpus/Input_Data/policy/9_(POS)policy_197.txt', 'Corpus/Input_Data/policy/9_(POS)policy_198.txt', 'Corpus/Input_Data/policy/9_(POS)policy_199.txt', 'Corpus/Input_Data/policy/9_(POS)policy_200.txt', 'Corpus/Input_Data/policy/9_(POS)policy_201.txt', 'Corpus/Input_Data/policy/9_(POS)policy_202.txt', 'Corpus/Input_Data/policy/9_(POS)policy_203.txt', 'Corpus/Input_Data/policy/9_(POS)policy_204.txt', 'Corpus/Input_Data/policy/9_(POS)policy_205.txt', 'Corpus/Input_Data/policy/9_(POS)policy_206.txt', 'Corpus/Input_Data/policy/9_(POS)policy_207.txt', 'Corpus/Input_Data/policy/9_(POS)policy_208.txt', 'Corpus/Input_Data/policy/9_(POS)policy_209.txt', 'Corpus/Input_Data/policy/9_(POS)policy_210.txt', 'Corpus/Input_Data/policy/9_(POS)policy_211.txt', 'Corpus/Input_Data/policy/9_(POS)policy_212.txt', 'Corpus/Input_Data/policy/9_(POS)policy_213.txt', 'Corpus/Input_Data/policy/9_(POS)policy_214.txt', 'Corpus/Input_Data/policy/9_(POS)policy_215.txt', 'Corpus/Input_Data/policy/9_(POS)policy_216.txt', 'Corpus/Input_Data/policy/9_(POS)policy_217.txt', 'Corpus/Input_Data/policy/9_(POS)policy_218.txt', 'Corpus/Input_Data/policy/9_(POS)policy_219.txt', 'Corpus/Input_Data/policy/9_(POS)policy_220.txt', 'Corpus/Input_Data/policy/9_(POS)policy_221.txt', 'Corpus/Input_Data/policy/9_(POS)policy_222.txt', 'Corpus/Input_Data/policy/9_(POS)policy_223.txt', 'Corpus/Input_Data/policy/9_(POS)policy_224.txt', 'Corpus/Input_Data/policy/9_(POS)policy_225.txt', 'Corpus/Input_Data/policy/9_(POS)policy_226.txt', 'Corpus/Input_Data/policy/9_(POS)policy_227.txt', 'Corpus/Input_Data/policy/9_(POS)policy_228.txt', 'Corpus/Input_Data/policy/9_(POS)policy_229.txt', 'Corpus/Input_Data/policy/9_(POS)policy_230.txt', 'Corpus/Input_Data/policy/9_(POS)policy_231.txt', 'Corpus/Input_Data/policy/9_(POS)policy_232.txt', 'Corpus/Input_Data/policy/9_(POS)policy_233.txt', 'Corpus/Input_Data/policy/9_(POS)policy_234.txt', 'Corpus/Input_Data/policy/9_(POS)policy_235.txt', 'Corpus/Input_Data/policy/9_(POS)policy_236.txt', 'Corpus/Input_Data/policy/9_(POS)policy_237.txt', 'Corpus/Input_Data/policy/9_(POS)policy_238.txt', 'Corpus/Input_Data/policy/9_(POS)policy_239.txt', 'Corpus/Input_Data/policy/9_(POS)policy_240.txt', 'Corpus/Input_Data/policy/9_(POS)policy_241.txt', 'Corpus/Input_Data/policy/9_(POS)policy_242.txt', 'Corpus/Input_Data/policy/9_(POS)policy_243.txt', 'Corpus/Input_Data/policy/9_(POS)policy_244.txt', 'Corpus/Input_Data/policy/9_(POS)policy_245.txt', 'Corpus/Input_Data/policy/9_(POS)policy_246.txt', 'Corpus/Input_Data/policy/9_(POS)policy_247.txt', 'Corpus/Input_Data/policy/9_(POS)policy_248.txt', 'Corpus/Input_Data/policy/9_(POS)policy_249.txt', 'Corpus/Input_Data/policy/9_(POS)policy_250.txt', 'Corpus/Input_Data/policy/9_(POS)policy_251.txt', 'Corpus/Input_Data/policy/9_(POS)policy_252.txt'], ['Corpus/Input_Data/society/9_(POS)society_1.txt', 'Corpus/Input_Data/society/9_(POS)society_2.txt', 'Corpus/Input_Data/society/9_(POS)society_3.txt', 'Corpus/Input_Data/society/9_(POS)society_4.txt', 'Corpus/Input_Data/society/9_(POS)society_5.txt', 'Corpus/Input_Data/society/9_(POS)society_6.txt', 'Corpus/Input_Data/society/9_(POS)society_7.txt', 'Corpus/Input_Data/society/9_(POS)society_8.txt', 'Corpus/Input_Data/society/9_(POS)society_9.txt', 'Corpus/Input_Data/society/9_(POS)society_10.txt', 'Corpus/Input_Data/society/9_(POS)society_11.txt', 'Corpus/Input_Data/society/9_(POS)society_12.txt', 'Corpus/Input_Data/society/9_(POS)society_13.txt', 'Corpus/Input_Data/society/9_(POS)society_14.txt', 'Corpus/Input_Data/society/9_(POS)society_15.txt', 'Corpus/Input_Data/society/9_(POS)society_16.txt', 'Corpus/Input_Data/society/9_(POS)society_17.txt', 'Corpus/Input_Data/society/9_(POS)society_18.txt', 'Corpus/Input_Data/society/9_(POS)society_19.txt', 'Corpus/Input_Data/society/9_(POS)society_20.txt', 'Corpus/Input_Data/society/9_(POS)society_21.txt', 'Corpus/Input_Data/society/9_(POS)society_22.txt', 'Corpus/Input_Data/society/9_(POS)society_23.txt', 'Corpus/Input_Data/society/9_(POS)society_24.txt', 'Corpus/Input_Data/society/9_(POS)society_25.txt', 'Corpus/Input_Data/society/9_(POS)society_26.txt', 'Corpus/Input_Data/society/9_(POS)society_27.txt', 'Corpus/Input_Data/society/9_(POS)society_28.txt', 'Corpus/Input_Data/society/9_(POS)society_29.txt', 'Corpus/Input_Data/society/9_(POS)society_30.txt', 'Corpus/Input_Data/society/9_(POS)society_31.txt', 'Corpus/Input_Data/society/9_(POS)society_32.txt', 'Corpus/Input_Data/society/9_(POS)society_33.txt', 'Corpus/Input_Data/society/9_(POS)society_34.txt', 'Corpus/Input_Data/society/9_(POS)society_35.txt', 'Corpus/Input_Data/society/9_(POS)society_36.txt', 'Corpus/Input_Data/society/9_(POS)society_37.txt', 'Corpus/Input_Data/society/9_(POS)society_38.txt', 'Corpus/Input_Data/society/9_(POS)society_39.txt', 'Corpus/Input_Data/society/9_(POS)society_40.txt', 'Corpus/Input_Data/society/9_(POS)society_41.txt', 'Corpus/Input_Data/society/9_(POS)society_42.txt', 'Corpus/Input_Data/society/9_(POS)society_43.txt', 'Corpus/Input_Data/society/9_(POS)society_44.txt', 'Corpus/Input_Data/society/9_(POS)society_45.txt', 'Corpus/Input_Data/society/9_(POS)society_46.txt', 'Corpus/Input_Data/society/9_(POS)society_47.txt', 'Corpus/Input_Data/society/9_(POS)society_48.txt', 'Corpus/Input_Data/society/9_(POS)society_49.txt', 'Corpus/Input_Data/society/9_(POS)society_50.txt', 'Corpus/Input_Data/society/9_(POS)society_51.txt', 'Corpus/Input_Data/society/9_(POS)society_52.txt', 'Corpus/Input_Data/society/9_(POS)society_53.txt', 'Corpus/Input_Data/society/9_(POS)society_54.txt', 'Corpus/Input_Data/society/9_(POS)society_55.txt', 'Corpus/Input_Data/society/9_(POS)society_56.txt', 'Corpus/Input_Data/society/9_(POS)society_57.txt', 'Corpus/Input_Data/society/9_(POS)society_58.txt', 'Corpus/Input_Data/society/9_(POS)society_59.txt', 'Corpus/Input_Data/society/9_(POS)society_60.txt', 'Corpus/Input_Data/society/9_(POS)society_61.txt', 'Corpus/Input_Data/society/9_(POS)society_62.txt', 'Corpus/Input_Data/society/9_(POS)society_63.txt', 'Corpus/Input_Data/society/9_(POS)society_64.txt', 'Corpus/Input_Data/society/9_(POS)society_65.txt', 'Corpus/Input_Data/society/9_(POS)society_66.txt', 'Corpus/Input_Data/society/9_(POS)society_67.txt', 'Corpus/Input_Data/society/9_(POS)society_68.txt', 'Corpus/Input_Data/society/9_(POS)society_69.txt', 'Corpus/Input_Data/society/9_(POS)society_70.txt', 'Corpus/Input_Data/society/9_(POS)society_71.txt', 'Corpus/Input_Data/society/9_(POS)society_72.txt', 'Corpus/Input_Data/society/9_(POS)society_73.txt', 'Corpus/Input_Data/society/9_(POS)society_74.txt', 'Corpus/Input_Data/society/9_(POS)society_75.txt', 'Corpus/Input_Data/society/9_(POS)society_76.txt', 'Corpus/Input_Data/society/9_(POS)society_77.txt', 'Corpus/Input_Data/society/9_(POS)society_78.txt', 'Corpus/Input_Data/society/9_(POS)society_79.txt', 'Corpus/Input_Data/society/9_(POS)society_80.txt', 'Corpus/Input_Data/society/9_(POS)society_81.txt', 'Corpus/Input_Data/society/9_(POS)society_82.txt', 'Corpus/Input_Data/society/9_(POS)society_83.txt', 'Corpus/Input_Data/society/9_(POS)society_84.txt', 'Corpus/Input_Data/society/9_(POS)society_85.txt', 'Corpus/Input_Data/society/9_(POS)society_86.txt', 'Corpus/Input_Data/society/9_(POS)society_87.txt', 'Corpus/Input_Data/society/9_(POS)society_88.txt', 'Corpus/Input_Data/society/9_(POS)society_89.txt', 'Corpus/Input_Data/society/9_(POS)society_90.txt', 'Corpus/Input_Data/society/9_(POS)society_91.txt', 'Corpus/Input_Data/society/9_(POS)society_92.txt', 'Corpus/Input_Data/society/9_(POS)society_93.txt', 'Corpus/Input_Data/society/9_(POS)society_94.txt', 'Corpus/Input_Data/society/9_(POS)society_95.txt', 'Corpus/Input_Data/society/9_(POS)society_96.txt', 'Corpus/Input_Data/society/9_(POS)society_97.txt', 'Corpus/Input_Data/society/9_(POS)society_98.txt', 'Corpus/Input_Data/society/9_(POS)society_99.txt', 'Corpus/Input_Data/society/9_(POS)society_100.txt', 'Corpus/Input_Data/society/9_(POS)society_101.txt', 'Corpus/Input_Data/society/9_(POS)society_102.txt', 'Corpus/Input_Data/society/9_(POS)society_103.txt', 'Corpus/Input_Data/society/9_(POS)society_104.txt', 'Corpus/Input_Data/society/9_(POS)society_105.txt', 'Corpus/Input_Data/society/9_(POS)society_106.txt', 'Corpus/Input_Data/society/9_(POS)society_107.txt', 'Corpus/Input_Data/society/9_(POS)society_108.txt', 'Corpus/Input_Data/society/9_(POS)society_109.txt', 'Corpus/Input_Data/society/9_(POS)society_110.txt', 'Corpus/Input_Data/society/9_(POS)society_111.txt', 'Corpus/Input_Data/society/9_(POS)society_112.txt', 'Corpus/Input_Data/society/9_(POS)society_113.txt', 'Corpus/Input_Data/society/9_(POS)society_114.txt', 'Corpus/Input_Data/society/9_(POS)society_115.txt', 'Corpus/Input_Data/society/9_(POS)society_116.txt', 'Corpus/Input_Data/society/9_(POS)society_117.txt', 'Corpus/Input_Data/society/9_(POS)society_118.txt', 'Corpus/Input_Data/society/9_(POS)society_119.txt', 'Corpus/Input_Data/society/9_(POS)society_120.txt', 'Corpus/Input_Data/society/9_(POS)society_121.txt', 'Corpus/Input_Data/society/9_(POS)society_122.txt', 'Corpus/Input_Data/society/9_(POS)society_123.txt', 'Corpus/Input_Data/society/9_(POS)society_124.txt', 'Corpus/Input_Data/society/9_(POS)society_125.txt', 'Corpus/Input_Data/society/9_(POS)society_126.txt', 'Corpus/Input_Data/society/9_(POS)society_127.txt', 'Corpus/Input_Data/society/9_(POS)society_128.txt', 'Corpus/Input_Data/society/9_(POS)society_129.txt', 'Corpus/Input_Data/society/9_(POS)society_130.txt', 'Corpus/Input_Data/society/9_(POS)society_131.txt', 'Corpus/Input_Data/society/9_(POS)society_132.txt', 'Corpus/Input_Data/society/9_(POS)society_133.txt', 'Corpus/Input_Data/society/9_(POS)society_134.txt', 'Corpus/Input_Data/society/9_(POS)society_135.txt', 'Corpus/Input_Data/society/9_(POS)society_136.txt', 'Corpus/Input_Data/society/9_(POS)society_137.txt', 'Corpus/Input_Data/society/9_(POS)society_138.txt', 'Corpus/Input_Data/society/9_(POS)society_139.txt', 'Corpus/Input_Data/society/9_(POS)society_140.txt', 'Corpus/Input_Data/society/9_(POS)society_141.txt', 'Corpus/Input_Data/society/9_(POS)society_142.txt', 'Corpus/Input_Data/society/9_(POS)society_143.txt', 'Corpus/Input_Data/society/9_(POS)society_144.txt', 'Corpus/Input_Data/society/9_(POS)society_145.txt', 'Corpus/Input_Data/society/9_(POS)society_146.txt', 'Corpus/Input_Data/society/9_(POS)society_147.txt', 'Corpus/Input_Data/society/9_(POS)society_148.txt', 'Corpus/Input_Data/society/9_(POS)society_149.txt', 'Corpus/Input_Data/society/9_(POS)society_150.txt', 'Corpus/Input_Data/society/9_(POS)society_151.txt', 'Corpus/Input_Data/society/9_(POS)society_152.txt', 'Corpus/Input_Data/society/9_(POS)society_153.txt', 'Corpus/Input_Data/society/9_(POS)society_154.txt', 'Corpus/Input_Data/society/9_(POS)society_155.txt', 'Corpus/Input_Data/society/9_(POS)society_156.txt', 'Corpus/Input_Data/society/9_(POS)society_157.txt', 'Corpus/Input_Data/society/9_(POS)society_158.txt', 'Corpus/Input_Data/society/9_(POS)society_159.txt', 'Corpus/Input_Data/society/9_(POS)society_160.txt', 'Corpus/Input_Data/society/9_(POS)society_161.txt', 'Corpus/Input_Data/society/9_(POS)society_162.txt', 'Corpus/Input_Data/society/9_(POS)society_163.txt', 'Corpus/Input_Data/society/9_(POS)society_164.txt', 'Corpus/Input_Data/society/9_(POS)society_165.txt', 'Corpus/Input_Data/society/9_(POS)society_166.txt', 'Corpus/Input_Data/society/9_(POS)society_167.txt', 'Corpus/Input_Data/society/9_(POS)society_168.txt', 'Corpus/Input_Data/society/9_(POS)society_169.txt', 'Corpus/Input_Data/society/9_(POS)society_170.txt', 'Corpus/Input_Data/society/9_(POS)society_171.txt', 'Corpus/Input_Data/society/9_(POS)society_172.txt', 'Corpus/Input_Data/society/9_(POS)society_173.txt', 'Corpus/Input_Data/society/9_(POS)society_174.txt', 'Corpus/Input_Data/society/9_(POS)society_175.txt', 'Corpus/Input_Data/society/9_(POS)society_176.txt', 'Corpus/Input_Data/society/9_(POS)society_177.txt', 'Corpus/Input_Data/society/9_(POS)society_178.txt', 'Corpus/Input_Data/society/9_(POS)society_179.txt', 'Corpus/Input_Data/society/9_(POS)society_180.txt', 'Corpus/Input_Data/society/9_(POS)society_181.txt', 'Corpus/Input_Data/society/9_(POS)society_182.txt', 'Corpus/Input_Data/society/9_(POS)society_183.txt', 'Corpus/Input_Data/society/9_(POS)society_184.txt', 'Corpus/Input_Data/society/9_(POS)society_185.txt', 'Corpus/Input_Data/society/9_(POS)society_186.txt', 'Corpus/Input_Data/society/9_(POS)society_187.txt', 'Corpus/Input_Data/society/9_(POS)society_188.txt', 'Corpus/Input_Data/society/9_(POS)society_189.txt', 'Corpus/Input_Data/society/9_(POS)society_190.txt', 'Corpus/Input_Data/society/9_(POS)society_191.txt', 'Corpus/Input_Data/society/9_(POS)society_192.txt', 'Corpus/Input_Data/society/9_(POS)society_193.txt', 'Corpus/Input_Data/society/9_(POS)society_194.txt', 'Corpus/Input_Data/society/9_(POS)society_195.txt', 'Corpus/Input_Data/society/9_(POS)society_196.txt', 'Corpus/Input_Data/society/9_(POS)society_197.txt', 'Corpus/Input_Data/society/9_(POS)society_198.txt', 'Corpus/Input_Data/society/9_(POS)society_199.txt', 'Corpus/Input_Data/society/9_(POS)society_200.txt', 'Corpus/Input_Data/society/9_(POS)society_201.txt', 'Corpus/Input_Data/society/9_(POS)society_202.txt', 'Corpus/Input_Data/society/9_(POS)society_203.txt', 'Corpus/Input_Data/society/9_(POS)society_204.txt', 'Corpus/Input_Data/society/9_(POS)society_205.txt', 'Corpus/Input_Data/society/9_(POS)society_206.txt', 'Corpus/Input_Data/society/9_(POS)society_207.txt', 'Corpus/Input_Data/society/9_(POS)society_208.txt', 'Corpus/Input_Data/society/9_(POS)society_209.txt', 'Corpus/Input_Data/society/9_(POS)society_210.txt', 'Corpus/Input_Data/society/9_(POS)society_211.txt', 'Corpus/Input_Data/society/9_(POS)society_212.txt', 'Corpus/Input_Data/society/9_(POS)society_213.txt', 'Corpus/Input_Data/society/9_(POS)society_214.txt', 'Corpus/Input_Data/society/9_(POS)society_215.txt', 'Corpus/Input_Data/society/9_(POS)society_216.txt', 'Corpus/Input_Data/society/9_(POS)society_217.txt', 'Corpus/Input_Data/society/9_(POS)society_218.txt', 'Corpus/Input_Data/society/9_(POS)society_219.txt', 'Corpus/Input_Data/society/9_(POS)society_220.txt', 'Corpus/Input_Data/society/9_(POS)society_221.txt', 'Corpus/Input_Data/society/9_(POS)society_222.txt', 'Corpus/Input_Data/society/9_(POS)society_223.txt', 'Corpus/Input_Data/society/9_(POS)society_224.txt', 'Corpus/Input_Data/society/9_(POS)society_225.txt', 'Corpus/Input_Data/society/9_(POS)society_226.txt', 'Corpus/Input_Data/society/9_(POS)society_227.txt', 'Corpus/Input_Data/society/9_(POS)society_228.txt', 'Corpus/Input_Data/society/9_(POS)society_229.txt', 'Corpus/Input_Data/society/9_(POS)society_230.txt', 'Corpus/Input_Data/society/9_(POS)society_231.txt', 'Corpus/Input_Data/society/9_(POS)society_232.txt', 'Corpus/Input_Data/society/9_(POS)society_233.txt', 'Corpus/Input_Data/society/9_(POS)society_234.txt', 'Corpus/Input_Data/society/9_(POS)society_235.txt', 'Corpus/Input_Data/society/9_(POS)society_236.txt', 'Corpus/Input_Data/society/9_(POS)society_237.txt', 'Corpus/Input_Data/society/9_(POS)society_238.txt', 'Corpus/Input_Data/society/9_(POS)society_239.txt', 'Corpus/Input_Data/society/9_(POS)society_240.txt', 'Corpus/Input_Data/society/9_(POS)society_241.txt', 'Corpus/Input_Data/society/9_(POS)society_242.txt', 'Corpus/Input_Data/society/9_(POS)society_243.txt', 'Corpus/Input_Data/society/9_(POS)society_244.txt', 'Corpus/Input_Data/society/9_(POS)society_245.txt', 'Corpus/Input_Data/society/9_(POS)society_246.txt', 'Corpus/Input_Data/society/9_(POS)society_247.txt', 'Corpus/Input_Data/society/9_(POS)society_248.txt', 'Corpus/Input_Data/society/9_(POS)society_249.txt', 'Corpus/Input_Data/society/9_(POS)society_250.txt', 'Corpus/Input_Data/society/9_(POS)society_251.txt', 'Corpus/Input_Data/society/9_(POS)society_252.txt', 'Corpus/Input_Data/society/9_(POS)society_253.txt', 'Corpus/Input_Data/society/9_(POS)society_254.txt', 'Corpus/Input_Data/society/9_(POS)society_255.txt', 'Corpus/Input_Data/society/9_(POS)society_256.txt', 'Corpus/Input_Data/society/9_(POS)society_257.txt', 'Corpus/Input_Data/society/9_(POS)society_258.txt', 'Corpus/Input_Data/society/9_(POS)society_259.txt', 'Corpus/Input_Data/society/9_(POS)society_260.txt', 'Corpus/Input_Data/society/9_(POS)society_261.txt', 'Corpus/Input_Data/society/9_(POS)society_262.txt', 'Corpus/Input_Data/society/9_(POS)society_263.txt', 'Corpus/Input_Data/society/9_(POS)society_264.txt', 'Corpus/Input_Data/society/9_(POS)society_265.txt', 'Corpus/Input_Data/society/9_(POS)society_266.txt', 'Corpus/Input_Data/society/9_(POS)society_267.txt', 'Corpus/Input_Data/society/9_(POS)society_268.txt', 'Corpus/Input_Data/society/9_(POS)society_269.txt', 'Corpus/Input_Data/society/9_(POS)society_270.txt', 'Corpus/Input_Data/society/9_(POS)society_271.txt', 'Corpus/Input_Data/society/9_(POS)society_272.txt', 'Corpus/Input_Data/society/9_(POS)society_273.txt', 'Corpus/Input_Data/society/9_(POS)society_274.txt', 'Corpus/Input_Data/society/9_(POS)society_275.txt', 'Corpus/Input_Data/society/9_(POS)society_276.txt', 'Corpus/Input_Data/society/9_(POS)society_277.txt', 'Corpus/Input_Data/society/9_(POS)society_278.txt', 'Corpus/Input_Data/society/9_(POS)society_279.txt', 'Corpus/Input_Data/society/9_(POS)society_280.txt', 'Corpus/Input_Data/society/9_(POS)society_281.txt', 'Corpus/Input_Data/society/9_(POS)society_282.txt', 'Corpus/Input_Data/society/9_(POS)society_283.txt', 'Corpus/Input_Data/society/9_(POS)society_284.txt', 'Corpus/Input_Data/society/9_(POS)society_285.txt', 'Corpus/Input_Data/society/9_(POS)society_286.txt', 'Corpus/Input_Data/society/9_(POS)society_287.txt', 'Corpus/Input_Data/society/9_(POS)society_288.txt', 'Corpus/Input_Data/society/9_(POS)society_289.txt', 'Corpus/Input_Data/society/9_(POS)society_290.txt', 'Corpus/Input_Data/society/9_(POS)society_291.txt', 'Corpus/Input_Data/society/9_(POS)society_292.txt', 'Corpus/Input_Data/society/9_(POS)society_293.txt', 'Corpus/Input_Data/society/9_(POS)society_294.txt', 'Corpus/Input_Data/society/9_(POS)society_295.txt', 'Corpus/Input_Data/society/9_(POS)society_296.txt', 'Corpus/Input_Data/society/9_(POS)society_297.txt', 'Corpus/Input_Data/society/9_(POS)society_298.txt', 'Corpus/Input_Data/society/9_(POS)society_299.txt', 'Corpus/Input_Data/society/9_(POS)society_300.txt', 'Corpus/Input_Data/society/9_(POS)society_301.txt', 'Corpus/Input_Data/society/9_(POS)society_302.txt', 'Corpus/Input_Data/society/9_(POS)society_303.txt', 'Corpus/Input_Data/society/9_(POS)society_304.txt', 'Corpus/Input_Data/society/9_(POS)society_305.txt', 'Corpus/Input_Data/society/9_(POS)society_306.txt', 'Corpus/Input_Data/society/9_(POS)society_307.txt', 'Corpus/Input_Data/society/9_(POS)society_308.txt', 'Corpus/Input_Data/society/9_(POS)society_309.txt', 'Corpus/Input_Data/society/9_(POS)society_310.txt', 'Corpus/Input_Data/society/9_(POS)society_311.txt', 'Corpus/Input_Data/society/9_(POS)society_312.txt', 'Corpus/Input_Data/society/9_(POS)society_313.txt', 'Corpus/Input_Data/society/9_(POS)society_314.txt', 'Corpus/Input_Data/society/9_(POS)society_315.txt', 'Corpus/Input_Data/society/9_(POS)society_316.txt', 'Corpus/Input_Data/society/9_(POS)society_317.txt', 'Corpus/Input_Data/society/9_(POS)society_318.txt', 'Corpus/Input_Data/society/9_(POS)society_319.txt', 'Corpus/Input_Data/society/9_(POS)society_320.txt', 'Corpus/Input_Data/society/9_(POS)society_321.txt', 'Corpus/Input_Data/society/9_(POS)society_322.txt', 'Corpus/Input_Data/society/9_(POS)society_323.txt', 'Corpus/Input_Data/society/9_(POS)society_324.txt', 'Corpus/Input_Data/society/9_(POS)society_325.txt', 'Corpus/Input_Data/society/9_(POS)society_326.txt', 'Corpus/Input_Data/society/9_(POS)society_327.txt']]\n"
     ]
    }
   ],
   "source": [
    "# search directories :: path ex) Corpus/Input_Data/child\n",
    "def search_dir(root_dir, dirs):\n",
    "    for file in os.listdir(root_dir):\n",
    "        path = os.path.join(root_dir, file)\n",
    "        if os.path.isdir(path):\n",
    "            dirs.append(path)\n",
    "    dirs.sort()\n",
    "\n",
    "# search files :: path ex) Corpus/Input_Data/child/13_(POS)child_1.txt\n",
    "def search_files(dirname, files):\n",
    "    try:\n",
    "        filenames = os.listdir(dirname)\n",
    "        filenames = natsort.natsorted(filenames)\n",
    "        for filename in filenames:\n",
    "            full_filename = os.path.join(dirname, filename)\n",
    "            if os.path.isdir(full_filename):\n",
    "                search_files(full_filename, files)\n",
    "            else:\n",
    "                ext = os.path.splitext(full_filename)[-1]\n",
    "                if ext == '.txt':\n",
    "                    files.append(full_filename)\n",
    "    except PermissionError:\n",
    "        pass\n",
    "\n",
    "root = 'Corpus/Input_Data'\n",
    "dir_path = []\n",
    "\n",
    "search_dir(root, dir_path)\n",
    "\n",
    "input_data = []\n",
    "for i in range(0,len(dir_path)):\n",
    "    data_path = []\n",
    "    search_files(dir_path[i], data_path)\n",
    "    input_data.append(data_path)\n",
    "\n",
    "print(dir_path)\n",
    "print(input_data)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "\n",
    "## Create Folder"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 652,
   "outputs": [],
   "source": [
    "# create folder\n",
    "def create_folder(directory):\n",
    "    try:\n",
    "        if not os.path.exists(directory):\n",
    "            os.makedirs(directory)\n",
    "            print(\"OK\")\n",
    "    except OSError:\n",
    "        print ('Error: Creating directory. ' +  directory)\n",
    "\n",
    "student_id = '202035535_leejiyun'\n",
    "for i in range(0, len(dir_path)):\n",
    "    temp_path = dir_path[i].split('/')\n",
    "    path_key = temp_path[1]+ '/' + temp_path[2]\n",
    "    create_folder(student_id + '/' + str(path_key))"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "## 1.1\n",
    " - from Input_Data\n",
    "    1. Extract nouns from input data\n",
    "    2. Extraction of 5000 least frequently\n",
    "    3. calculate TF\n",
    "    4. calculate IDF\n",
    "    5. calculate TF-IDF\n",
    "    6. Write files"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### 1. Extract nouns from input data & 2. Extraction of 5000 least frequently"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 653,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['ⓒ샘/NNG', '가격/NNG', '가계/NNG', '가구/NNG', '가구소득/NNG', '가난/NNG', '가능/NNG', '가능성/NNG', '가동/NNG', '가명/NNG', '가방/NNG', '가사/NNG', '가산/NNG', '가수/NNG', '가슴/NNG', '가시/NNG', '가운데/NNG', '가을/NNG', '가을철/NNG', '가이드/NNG', '가이드라인/NNG', '가이드북/NNG', '가입/NNG', '가입자/NNG', '가정/NNG', '가정산소치료서비스/NNG', '가정폭력/NNG', '가족/NNG', '가족여행/NNG', '가중/NNG', '가짜/NNG', '가처분소득/NNG', '가치/NNG', '가칭/NNG', '가해자/NNG', '가해학생/NNG', '가혹행위/NNG', '각/NNG', '각각/NNG', '각계/NNG', '각국/NNG', '각급학교/NNG', '각오/NNG', '각자/NNG', '각종/NNG', '각지/NNG', '각층/NNG', '간격/NNG', '간담회/NNG', '간담회’/NNG', '간병/NNG', '간부/NNG', '간사/NNG', '간암/NNG', '간이/NNG', '간질/NNG', '간판/NNG', '간호사/NNG', '갈등/NNG', '감금/NNG', '감기/NNG', '감독/NNG', '감동/NNG', '감동적/NNG', '감면/NNG', '감사/NNG', '감사원/NNG', '감소/NNG', '감시/NNG', '감시단/NNG', '감염/NNG', '감염병/NNG', '감정/NNG', '값/NNG', '강/NNG', '강/NNP', '강간/NNG', '강남/NNP', '강남구/NNP', '강도/NNG', '강동구/NNP', '강릉/NNP', '강북/NNP', '강북구/NNP', '강북지역/NNG', '강사/NNG', '강서구/NNP', '강연/NNG', '강연자/NNG', '강요/NNG', '강원/NNP', '강원도/NNP', '강의/NNG', '강의내용/NNG', '강점/NNG', '강제/NNG', '강제추행/NNG', '강화/NNG', '개/NNG', '개관식/NNG', '개념/NNG', '개막/NNG', '개막식/NNG', '개발/NNG', '개발원/NNG', '개별/NNG', '개별적/NNG', '개선/NNG', '개선방안/NNG', '개선책/NNG', '개설/NNG', '개성/NNG', '개소/NNG', '개소식/NNG', '개인/NNG', '개인별/NNG', '개인적/NNG', '개인전/NNG', '개정/NNG', '개정안/NNG', '개정안’/NNG', '개조/NNG', '개최/NNG', '개통/NNG', '개편/NNG', '개혁/NNG', '개회식/NNG', '객관적/NNG', '객원기자/NNG', '갱신기간/NNG', '거동/NNG', '거리/NNG', '거리노숙/NNG', '거리생활/NNG', '거부/NNG', '거세/NNG', '거소투표/NNG', '거액/NNG', '거약자/NNG', '거점/NNG', '거점투쟁/NNG', '거제/NNP', '거주/NNG', '거주시설/NNG', '거주자/NNG', '거주지/NNG', '거짓/NNG', '걱정/NNG', '건강/NNG', '건강검진/NNG', '건강관리/NNG', '건강보험/NNG', '건강보험공단/NNG', '건강보험료/NNG', '건강보험제도/NNG', '건강보험증/NNG', '건강상태/NNG', '건강증진/NNG', '건립/NNG', '건물/NNG', '건보/NNG', '건보공단/NNG', '건수/NNG', '건의/NNG', '건전성/NNG', '건축/NNG', '건축비/NNG', '건축협의/NNG', '걸림돌/NNG', '걸음/NNG', '검사/NNG', '검색어/NNG', '검정고시/NNG', '검증/NNG', '검진/NNG', '검진요건/NNG', '검찰/NNG', '검찰청/NNG', '검토/NNG', '검토의견/NNG', '게스트/NNG', '게시판/NNG', '게임/NNG', '겨울/NNG', '겨울철/NNG', '격려/NNG', '격리/NNG', '격차/NNG', '결과/NNG', '결과”/NNG', '결과물/NNG', '결국/NNG', '결론/NNG', '결산/NNG', '결선/NNG', '결승/NNG', '결승전/NNG', '결승점/NNG', '결실/NNG', '결심/NNG', '결원/NNG', '결의/NNG', '결의대회/NNG', '결의대회’/NNG', '결정/NNG', '결정권/NNG', '결함/NNG', '결핵/NNG', '결혼/NNG', '경/NNG', '경각심/NNG', '경감/NNG', '경계/NNG', '경계석/NNG', '경과/NNG', '경기/NNG', '경기/NNP', '경기도/NNP', '경기도교육청/NNP', '경기일정/NNG', '경기장/NNG', '경기지청/NNG', '경기침체/NNG', '경남/NNP', '경남도/NNP', '경력/NNG', '경력자/NNG', '경로/NNG', '경로당/NNG', '경북/NNP', '경북도/NNP', '경비/NNG', '경사/NNG', '경사로/NNG', '경상대/NNP', '경선/NNG', '경영/NNG', '경우/NNG', '경위/NNG', '경쟁/NNG', '경쟁력/NNG', '경쟁률/NNG', '경제/NNG', '경제민주화/NNG', '경제적/NNG', '경제활동/NNG', '경증/NNG', '경증장애인/NNG', '경찰/NNG', '경찰관/NNG', '경찰서/NNG', '경찰청/NNG', '경향/NNG', '경험/NNG', '경호원/NNG', '곁/NNG', '계기/NNG', '계단/NNG', '계류/NNG', '계산/NNG', '계속적/NNG', '계약/NNG', '계열사/NNG', '계절/NNG', '계층/NNG', '계획/NNG', '계획안/NNG', '고/NNG', '고/NNP', '고개/NNG', '고객/NNG', '고갯길/NNG', '고교/NNG', '고난/NNG', '고등/NNG', '고등교육/NNG', '고등법원/NNG', '고등학교/NNG', '고등학생/NNG', '고려/NNG', '고령자/NNG', '고령화/NNG', '고문/NNG', '고미숙/NNP', '고민/NNG', '고발/NNG', '고발장/NNG', '고법/NNG', '고비/NNG', '고사장/NNG', '고소/NNG', '고소득/NNG', '고속도로/NNG', '고시/NNG', '고액/NNG', '고양/NNG', '고양시/NNP', '고양어울림누리체육관/NNP', '고양종합운/NNP', '고용/NNG', '고용개발/NNG', '고용노동부/NNG', '고용률/NNG', '고용부/NNG', '고용부담금/NNG', '고용율/NNG', '고용의무/NNG', '고용장려금/NNG', '고용주/NNG', '고용확대/NNG', '고용활성화/NNG', '고위급회의/NNG', '고의/NNG', '고장/NNG', '고정/NNG', '고정관념/NNG', '고정식/NNG', '고졸/NNG', '고취/NNG', '고통/NNG', '고혈압/NNG', '고희숙/NNP', '곡/NNG', '골드/NNG', '골방/NNG', '골볼/NNG', '골자/NNG', '곳/NNG', '곳곳/NNG', '공/NNG', '공간/NNG', '공감/NNG', '공감대/NNG', '공개/NNG', '공격/NNG', '공고문/NNG', '공공/NNG', '공공건물/NNG', '공공기관/NNG', '공공단체/NNG', '공공성/NNG', '공공시설/NNG', '공공의료기관/NNG', '공공임대/NNG', '공공임대주택/NNG', '공공장소/NNG', '공공주택/NNG', '공급/NNG', '공급자/NNG', '공기/NNG', '공기권총/NNG', '공기업/NNG', '공단/NNG', '공대위/NNP', '공동/NNG', '공동대책위원회/NNG', '공동대표/NNG', '공동체/NNG', '공동취재/NNG', '공동행동/NNG', '공로/NNG', '공론/NNG', '공립/NNG', '공립학교/NNG', '공모/NNG', '공모전/NNG', '공무원/NNG', '공문/NNG', '공방/NNG', '공부/NNG', '공분/NNG', '공사/NNG', '공소시효/NNG', '공식/NNG', '공식일정/NNG', '공식적/NNG', '공안부/NNG', '공약/NNG', '공약’/NNG', '공연/NNG', '공연장/NNG', '공원/NNG', '공유/NNG', '공익/NNG', '공장/NNG', '공적/NNG', '공정성/NNG', '공제/NNG', '공존/NNG', '공주/NNP', '공중이용시설/NNG', '공중화장실/NNG', '공직/NNG', '공직선거법/NNG', '공채/NNG', '공천/NNG', '공천로비/NNG', '공청회/NNG', '공통/NNG', '공통과정/NNG', '공통적/NNG', '공투단/NNG', '공판/NNG', '공포/NNG', '공항/NNG', '공휴일/NNG', '과거/NNG', '과목/NNG', '과밀학급/NNG', '과밀화/NNG', '과세/NNG', '과실/NNG', '과일/NNG', '과장/NNG', '과정/NNG', '과제/NNG', '과제”/NNG', '과징금/NNG', '과천/NNP', '과태료/NNG', '관/NNG', '관객/NNG', '관계/NNG', '관계부처/NNG', '관계자/NNG', '관광/NNG', '관내/NNG', '관람/NNG', '관련/NNG', '관련기관/NNG', '관련단체/NNG', '관련자/NNG', '관리/NNG', '관리감독/NNG', '관리자/NNG', '관세청/NNG', '관심/NNG', '관심사/NNG', '관왕/NNG', '관장/NNG', '관점/NNG', '관중/NNG', '관철/NNG', '관할/NNG', '광고/NNG', '광명시/NNP', '광산구/NNP', '광역시/NNG', '광장/NNG', '광주/NNP', '광주광역시/NNP', '광주송정역/NNP', '광주시/NNP', '광주인화학교/NNP', '광화문/NNP', '광화문역/NNP', '교/NNG', '교감/NNG', '교과부/NNG', '교류/NNG', '교부/NNG', '교사/NNG', '교수/NNG', '교실/NNG', '교원/NNG', '교육/NNG', '교육감/NNG', '교육과정/NNG', '교육과학기술부/NNG', '교육과학기술위원회/NNG', '교육권/NNG', '교육권연대/NNG', '교육기관/NNG', '교육비/NNG', '교육장/NNG', '교육적/NNG', '교육지원/NNG', '교육청/NNG', '교육환경/NNG', '교장/NNG', '교재/NNG', '교직원/NNG', '교체/NNG', '교통/NNG', '교통비/NNG', '교통사고/NNG', '교통수단/NNG', '교통약자/NNG', '교회/NNG', '구/NNG', '구간/NNG', '구간대원/NNG', '구근호/NNP', '구글/NNG', '구매/NNG', '구매비율/NNG', '구멍/NNG', '구범/NNG', '구분/NNG', '구비추가지원/NNG', '구상/NNG', '구성/NNG', '구성원/NNG', '구속/NNG', '구역/NNG', '구의회/NNG', '구입/NNG', '구장/NNG', '구제/NNG', '구조/NNG', '구조적/NNG', '구청/NNG', '구청장/NNG', '구체적/NNG', '구축/NNG', '구호/NNG', '국/NNG', '국가/NNG', '국가공무원/NNG', '국가구/NNG', '국가기관/NNG', '국가대표/NNG', '국가보고서/NNG', '국가보훈처/NNG', '국가유공자/NNG', '국가인권위원회/NNG', '국가자격/NNG', '국가자격증/NNG', '국가줄기세포은행/NNP', '국고/NNG', '국고보조금/NNG', '국공립/NNG', '국내/NNG', '국내외/NNG', '국립대병원/NNG', '국립보건연구원/NNP', '국립장애인도서관/NNG', '국립재활원/NNP', '국립중앙도서관/NNP', '국립중앙의료원/NNP', '국립특수교육원/NNP', '국무총리/NNG', '국무회의/NNG', '국민/NNG', '국민건강보험/NNG', '국민건강보험공단/NNG', '국민건강보험법/NNG', '국민기초/NNP', '국민기초생활보장법/NNG', '국민여러/NNG', '국민연금/NNG', '국민연금공단/NNG', '국민연금공단/NNP', '국민연금법/NNG', '국민은행/NNP', '국민적/NNG', '국민주택기금/NNG', '국민행복/NNG', '국방부/NNG', '국비/NNG', '국선변호사/NNG', '국세청/NNG', '국악기/NNG', '국외/NNG', '국장/NNG', '국장애인교육권연대/NNG', '국장애인기능경기대회/NNG', '국장애인부모연대/NNG', '국장애인야학협의회/NNG', '국장애인차별철폐연대/NNG', '국장애인체육대회/NNG', '국장애인체전/NNG', '국적/NNG', '국정/NNG', '국정감사/NNG', '국정감사자료/NNG', '국제/NNG', '국제기구/NNG', '국제대회/NNG', '국제스포츠대회/NNG', '국제영화제/NNG', '국제장애어린이축제/NNG', '국제장애인연맹/NNG', '국제장애인올림픽위원회/NNG', '국제적/NNG', '국제협력/NNG', '국제회의장/NNG', '국토대장정/NNG', '국토대장정’/NNG', '국토부/NNG', '국토해양부/NNP', '국회/NNG', '국회의원/NNG', '군/NNG', '군수/NNG', '군인/NNG', '군청/NNG', '굿/NNG', '굿윌/NNG', '굿윌/NNP', '궁극적/NNG', '권/NNP', '권고/NNG', '권력/NNG', '권리/NNG', '권리구제/NNG', '권리보장/NNG', '권리옹호/NNP', '권순기/NNP', '권역/NNG', '권익/NNG', '권익보호/NNG', '권익옹호/NNG', '권익위/NNG', '권익증진/NNG', '권익향상/NNG', '권인자/NNP', '권총/NNG', '권한/NNG', '귀/NNG', '귓속/NNG', '규격/NNG', '규모/NNG', '규정/NNG', '규제/NNG', '규칙/NNG', '규탄/NNG', '균도/NNG', '균형/NNG', '그간/NNG', '그날/NNG', '그동안/NNG', '그때/NNG', '그랜드/NNG', '그룹/NNG', '그룹홈/NNG', '그리스/NNP', '그림/NNG', '극장/NNG', '근거/NNG', '근거규정/NNG', '근골격계/NNG', '근로/NNG', '근로기준법/NNG', '근로복지공단/NNG', '근로소득/NNG', '근로자/NNG', '근로지원/NNG', '근로지원서비스/NNG', '근로지원인/NNG', '근무/NNG', '근본적/NNG', '근육/NNG', '근절/NNG', '근처/NNG', '글/NNG', '금/NNG', '금년/NNG', '금메달/NNG', '금메달리스트/NNG', '금번/NNG', '금빛/NNG', '금액/NNG', '금연구역/NNG', '금융/NNG', '금융기관/NNG', '금융재산/NNG', '금융정보/NNG', '금융정보등제공동의서/NNG', '금일봉/NNG', '금정구/NNP', '금지/NNG', '금품/NNG', '금품갈취/NNG', '급/NNG', '급경사/NNG', '급식/NNG', '급식비/NNG', '급여/NNG', '급여량/NNG', '급여품목/NNG', '긍정적/NNG', '기/NNG', '기간/NNG', '기간동안/NNG', '기간제/NNG', '기계/NNG', '기관/NNG', '기구/NNG', '기금/NNG', '기기/NNG', '기념사진/NNG', '기념식/NNG', '기념촬영/NNG', '기능/NNG', '기능검사/NNG', '기능보강/NNG', '기능보강사업/NNG', '기능적/NNG', '기대/NNG', '기대감/NNG', '기도/NNG', '기득권/NNG', '기량/NNG', '기록/NNG', '기반/NNG', '기본/NNG', '기본권/NNG', '기본급여/NNG', '기본적/NNG', '기부/NNG', '기부금/NNG', '기부문화/NNG', '기분/NNG', '기쁨/NNG', '기사/NNG', '기숙사/NNG', '기술/NNG', '기억/NNG', '기업/NNG', '기업은행/NNP', '기업체/NNG', '기여/NNG', '기온/NNG', '기원/NNG', '기자/NNG', '기자회견/NNG', '기재/NNG', '기재부/NNG', '기적/NNG', '기조/NNG', '기조연설/NNG', '기존/NNG', '기종/NNG', '기준/NNG', '기준안/NNG', '기초/NNG', '기초급여/NNG', '기초노령연금/NNG', '기초법/NNG', '기초생활/NNG', '기초생활보장/NNG', '기초생활수급/NNG', '기초생활수급비/NNG', '기초생활수급자/NNG', '기초수급/NNG', '기초자치단체/NNG', '기침/NNG', '기타/NNG', '기타공공기관/NNG', '기표소/NNG', '기호/NNG', '기회/NNG', '기획/NNG', '기획재정부/NNG', '긴급/NNG', '긴급대책/NNG', '긴장/NNG', '길/NNG', '길’/NNG', '길거리/NNG', '길이/NNG', '김/NNP', '김경묵/NNP', '김규대/NNP', '김금래/NNP', '김기룡/NNP', '김대성/NNP', '김동기/NNP', '김두관/NNP', '김란숙/NNP', '김문수/NNP', '김선미/NNP', '김세식/NNP', '김세연/NNP', '김소연/NNP', '김수진/NNP', '김연정/NNP', '김영건/NNP', '김우남/NNP', '김정록/NNP', '김정하/NNP', '김주영/NNP', '김지은/NNP', '김철민/NNP', '김태호/NNP', '김포/NNP', '김한수/NNP', '김해시청/NNP', '김형식/NNP', '김호연/NNP', '김황식/NNP', '김효진/NNP', '꼼수/NNG', '꿈/NNG', '끈/NNG', '끝/NNG', '끼/NNG', '나경원/NNP', '나라/NNG', '나머지/NNG', '나무/NNG', '나사렛대학교/NNP', '나이/NNG', '나이키/NNP', '나중/NNG', '낙인/NNG', '난관/NNG', '난방/NNG', '난색/NNG', '날/NNG', '날씨/NNG', '날짜/NNG', '남/NNG', '남/NNP', '남녀/NNG', '남녀공용장애인화장실/NNG', '남녀비장애인화장실/NNG', '남녀장애인화장실/NNG', '남동생/NNG', '남매/NNG', '남북/NNP', '남산케이블카/NNP', '남성/NNG', '남성장애인/NNG', '남양주시/NNP', '남여/NNG', '남윤/NNP', '남윤인순/NNP', '남자/NNG', '남자비장애인화장실/NNG', '남자화장실/NNG', '남편/NNG', '납부/NNG', '납부기한/NNG', '납부증명서/NNG', '내/NNG', '내년/NNG', '내년도/NNG', '내달/NNG', '내리막길/NNG', '내부/NNG', '내부적/NNG', '내빈/NNG', '내역/NNG', '내외/NNG', '내용/NNG', '내일/NNG', '냄새/NNG', '넋/NNG', '네이버/NNP', '네트워크/NNG', '네티즌/NNG', '네팔/NNP', '네펠리/NNP', '넷플릭스/NNP', '녀/NNG', '노/NNP', '노고/NNG', '노년기/NNG', '노동/NNG', '노동권/NNG', '노동부/NNG', '노동시장/NNG', '노동자/NNG', '노래/NNG', '노력/NNG', '노령/NNG', '노령연금/NNG', '노메달/NNG', '노사/NNG', '노숙/NNG', '노숙농성/NNG', '노숙인/NNG', '노스/NNG', '노약자/NNG', '노원구/NNP', '노익상/NNP', '노인/NNG', '노인돌봄종합서비스/NNG', '노출/NNG', '노컷뉴/NNP', '노하우/NNG', '노후/NNG', '녹취/NNG', '녹화/NNG', '논란/NNG', '논리/NNG', '논문/NNG', '논의/NNG', '논평/NNG', '농/NNG', '농교육/NNG', '농구/NNG', '농성/NNG', '농아/NNG', '농아/NNP', '농아인/NNG', '농아인협회/NNG', '농아학생/NNG', '농어촌/NNG', '농인/NNG', '농학생/NNG', '높이/NNG', '뇌/NNG', '뇌병변/NNG', '뇌병변/NNP', '뇌병변장애/NNG', '뇌병변장애인/NNG', '뇌병변장애인인권협회/NNG', '뇌성마비/NNG', '뇌성마비장애인/NNG', '뇌염/NNG', '누나/NNG', '누리과정/NNG', '누리과정’/NNG', '누리홀/NNG', '눈/NNG', '눈길/NNG', '눈높이/NNG', '눈물/NNG', '뉴/NNG', '뉴스/NNG', '뉴욕/NNP', '뉴질랜드/NNP', '느낌/NNG', '늑대소년/NNG', '능력/NNG', '다/NNG', '다각적/NNG', '다리/NNG', '다목적홀/NNG', '다문화가정/NNG', '다솜이재단/NNG', '다수/NNG', '다양성/NNG', '다양화/NNG', '다운증후군/NNG', '다음/NNG', '다음달/NNG', '다음해/NNG', '다큐멘터리/NNG', '다행/NNG', '닥터헬기/NNG', '단/NNG', '단가/NNG', '단계/NNG', '단계적/NNG', '단기/NNG', '단기보호시설/NNG', '단독/NNG', '단백질/NNG', '단속/NNG', '단순/NNG', '단식/NNG', '단어/NNG', '단원/NNG', '단원구청/NNG', '단위/NNG', '단일화/NNG', '단장/NNG', '단점/NNG', '단체/NNG', '단체전/NNG', '단축/NNG', '달/NNG', '달동네/NNG', '담당/NNG', '담당자/NNG', '담당직원/NNG', '담배/NNG', '담보/NNG', '담임교사/NNG', '답/NNG', '답변/NNG', '당/NNG', '당국/NNG', '당뇨/NNG', '당뇨망막병증/NNG', '당뇨병/NNG', '당부/NNG', '당사/NNG', '당사자/NNG', '당선/NNG', '당선인/NNG', '당시/NNG', '당일/NNG', '당장/NNG', '당진시/NNP', '당초/NNG', '대/NNG', '대가/NNG', '대결/NNG', '대구/NNP', '대구대/NNP', '대구대학교/NNP', '대구치/NNG', '대규모/NNG', '대기/NNG', '대기시간/NNG', '대기업/NNG', '대다수/NNG', '대답/NNG', '대대적/NNG', '대도시/NNG', '대량/NNG', '대륙/NNG', '대리/NNG', '대리인/NNG', '대리투표/NNG', '대만/NNP', '대법원/NNG', '대변인/NNG', '대부분/NNG', '대비/NNG', '대사/NNG', '대상/NNG', '대상자/NNG', '대상학생/NNG', '대선/NNG', '대선공약/NNG', '대선연대/NNG', '대선장애인연대/NNG', '대선출마/NNG', '대선후보/NNG', '대신/NNG', '대안/NNG', '대여/NNG', '대열/NNG', '대왕/NNG', '대원/NNG', '대응/NNG', '대입/NNG', '대장/NNG', '대장정/NNG', '대전/NNG', '대전/NNP', '대전시/NNP', '대중/NNG', '대중교통/NNG', '대책/NNG', '대책마련/NNG', '대책위/NNG', '대처/NNG', '대출/NNG', '대통령/NNG', '대통령령/NNG', '대통령선거/NNG', '대표/NNG', '대표단/NNG', '대표선수단/NNG', '대표이사/NNG', '대표자/NNG', '대표적/NNG', '대표팀/NNG', '대학/NNG', '대학교/NNG', '대학생/NNG', '대학원/NNG', '대학입학/NNG', '대한/NNP', '대한민국/NNP', '대한신장학회/NNP', '대한장애인체육회/NNP', '대행/NNG', '대형/NNG', '대형마트/NNG', '대화/NNG', '대회/NNG', '대회기간/NNG', '댓글/NNG', '덕분/NNG', '데이비스/NNP', '데이터/NNG', '데이터베이스/NNG', '도/NNG', '도가니/NNG', '도경만/NNP', '도교육청/NNG', '도구/NNG', '도내/NNG', '도덕적/NNG', '도둑/NNG', '도로/NNG', '도로교통공단/NNG', '도보/NNG', '도봉구/NNP', '도서/NNG', '도서관/NNG', '도시/NNG', '도시지역/NNG', '도약/NNG', '도우미/NNG', '도움/NNG', '도입/NNG', '도입계획/NNG', '도입률/NNG', '도자기/NNG', '도전/NNG', '도중/NNG', '도지사/NNG', '도착/NNG', '독거/NNG', '독거장애인/NNG', '독립기관/NNG', '독립성/NNG', '독립적/NNG', '독서/NNG', '독일/NNP', '독자적/NNG', '독후감/NNG', '돈/NNG', '돌/NNG', '돌고래/NNG', '돌봄/NNG', '돌봄서비스/NNG', '돌입/NNG', '동/NNG', '동계/NNG', '동기/NNG', '동네/NNG', '동대문구/NNP', '동력/NNG', '동료/NNG', '동료상담/NNG', '동메달/NNG', '동물/NNG', '동물원/NNG', '동반/NNG', '동백열차/NNG', '동법/NNG', '동북아/NNP', '동사무소/NNG', '동상/NNG', '동생/NNG', '동석/NNG', '동시/NNG', '동시다발/NNG', '동안/NNG', '동영상/NNG', '동원홈푸드/NNP', '동의/NNG', '동의서/NNG', '동일/NNG', '동작구/NNP', '동장/NNG', '동정/NNG', '동주민센터/NNG', '동지/NNG', '동쪽/NNG', '돼지/NNG', '두려움/NNG', '뒤/NNG', '뒤쪽/NNG', '뒷면/NNG', '드라마/NNG', '등/NNG', '등급/NNG', '등록/NNG', '등록금/NNG', '등록등급/NNG', '등록면허세/NNG', '등록장애인/NNG', '등록제/NNG', '등받이/NNG', '디스에빌리티/NNG', '디자인/NNG', '디지털/NNG', '딸/NNG', '땀/NNG', '땀띠/NNG', '땅/NNG', '때/NNG', '땡볕/NNG', '떡/NNG', '뜻/NNG', '라디오/NNG', '러닝타임/NNG', '러시아/NNP', '런던/NNP', '레이스/NNG', '로더/NNP', '로맨스/NNG', '로보틱/NNG', '로보틱/NNP', '로비/NNG', '롯데마트/NNP', '룸/NNG', '리더십/NNG', '리모델링/NNG', '리모컨/NNG', '리커브/NNP', '리프트/NNG', '릴레이/NNG', '림주성/NNG', '림픽/NNG', '마다/NNG', '마디/NNG', '마라톤/NNG', '마련/NNG', '마무리/NNG', '마비/NNG', '마사회/NNG', '마을/NNG', '마음/NNG', '마지막/NNG', '마찬가지/NNG', '마찰/NNG', '마케도/NNP', '마케팅/NNG', '마크/NNG', '마포구/NNP', '막/NNG', '막바지/NNG', '만남/NNG', '만성질환/NNG', '만성콩팥병/NNG', '만성통증/NNG', '만약/NNG', '만전/NNG', '만족/NNG', '만족도/NNG', '말/NNG', '말문/NNG', '말씀/NNG', '맞벌이/NNG', '맞춤/NNG', '매뉴얼/NNG', '매스컴/NNG', '매장/NNG', '매점/NNG', '매칭펀드/NNG', '매트/NNG', '매표소/NNG', '맥/NNP', '머리/NNG', '먹구름/NNG', '메뉴/NNG', '메달/NNG', '메시지/NNG', '메아리/NNG', '메카/NNG', '메카/NNP', '메커니즘/NNG', '멤버/NNG', '면/NNG', '면담/NNG', '면담요청서/NNG', '면세/NNG', '면적/NNG', '면접/NNG', '면접시험/NNG', '면접캠프/NNG', '면제/NNG', '면허/NNG', '명/NNG', '명단/NNG', '명단공개/NNG', '명령/NNG', '명목/NNG', '명분/NNG', '명세서/NNG', '명시/NNG', '명예/NNG', '명주/NNP', '명칭/NNG', '모금/NNG', '모녀/NNG', '모니터/NNG', '모니터링/NNG', '모델/NNG', '모두/NNG', '모바일/NNG', '모범/NNG', '모색/NNG', '모성권/NNG', '모스크바/NNP', '모습/NNG', '모양/NNG', '모의평가/NNG', '모임/NNG', '모자/NNG', '모집/NNG', '모터/NNG', '모회사/NNG', '목/NNG', '목발/NNG', '목사/NNG', '목소리/NNG', '목숨/NNG', '목욕의자/NNG', '목원대학교/NNP', '목적/NNG', '목적지/NNG', '목표/NNG', '목표치/NNG', '몫/NNG', '몸/NNG', '몸싸움/NNG', '몽골/NNP', '무/NNG', '무가선트램/NNG', '무게/NNG', '무관심/NNG', '무기한/NNG', '무단/NNG', '무대/NNG', '무료/NNG', '무릎/NNG', '무리/NNG', '무산/NNG', '무상/NNG', '무상보육/NNG', '무소속/NNG', '무시/NNG', '무용지물/NNG', '무자격자/NNG', '무작정/NNG', '무장애/NNG', '무죄/NNG', '무지/NNG', '무혐의/NNG', '문/NNG', '문/NNP', '문경희/NNP', '문고리/NNG', '문광부/NNG', '문구/NNG', '문서/NNG', '문성혜/NNP', '문의/NNG', '문장/NNG', '문재/NNG', '문재인/NNG', '문재인/NNP', '문제/NNG', '문제”/NNG', '문제점/NNG', '문제지/NNG', '문턱/NNG', '문학/NNG', '문항/NNG', '문화/NNG', '문화산업/NNG', '문화예술/NNG', '문화적/NNG', '문화체육관광방송통신/NNP', '문화체육관광부/NNG', '문화특별보좌관/NNG', '물/NNG', '물가상승/NNG', '물거품/NNG', '물건/NNG', '물론/NNG', '물리적/NNG', '물살/NNG', '물의/NNG', '물줄기/NNG', '물질/NNG', '물체/NNG', '물품/NNG', '뮤지컬/NNG', '미/NNP', '미국/NNP', '미닫이/NNG', '미닫이문/NNG', '미달/NNG', '미래/NNG', '미만/NNG', '미비/NNG', '미비점/NNG', '미설치/NNG', '미성년자/NNG', '미소/NNG', '미술/NNG', '미술관/NNG', '미얀마/NNP', '미연/NNG', '미혼/NNG', '미흡/NNG', '민간/NNG', '민간기업/NNG', '민간단체/NNG', '민간부문/NNG', '민간위탁/NNG', '민병언/NNG', '민병언/NNP', '민영규/NNP', '민원/NNG', '민원국/NNG', '민원실/NNG', '민재/NNG', '민주당/NNP', '민주주의/NNG', '민주통합당/NNP', '민주화/NNG', '밑/NNG', '밑바닥/NNG', '바깥/NNG', '바다/NNG', '바닥/NNG', '바둑/NNG', '바둑판/NNG', '바람/NNG', '바리스타/NNG', '바바라/NNP', '바우처/NNP', '바퀴/NNG', '바탕/NNG', '박/NNP', '박경석/NNP', '박근혜/NNP', '박김영희/NNP', '박물관/NNG', '박사/NNG', '박성호/NNP', '박세균/NNP', '박세진/NNP', '박수/NNG', '박스/NNG', '박원석/NNP', '박원순/NNP', '박재희/NNP', '박정선/NNP', '박종태/NNG', '박종태/NNP', '박준영/NNP', '박지우/NNP', '박진영/NNP', '박탈/NNG', '박홍구/NNP', '밖/NNG', '반/NNG', '반기문/NNP', '반대/NNG', '반면/NNG', '반발/NNG', '반복/NNG', '반시설/NNG', '반영/NNG', '반응/NNG', '반장/NNG', '반죽/NNG', '발/NNG', '발간/NNG', '발견/NNG', '발굴/NNG', '발급/NNG', '발급기관/NNG', '발달/NNG', '발달장애/NNG', '발달장애국/NNG', '발달장애인/NNG', '발달장애인법/NNG', '발달장애인법안/NNG', '발달장애청년/NNG', '발대식/NNG', '발생/NNG', '발언/NNG', '발의/NNG', '발작/NNG', '발전/NNG', '발전방안/NNG', '발제/NNG', '발진/NNG', '발표/NNG', '밤/NNG', '밥/NNG', '방/NNG', '방과후/NNG', '방귀희/NNP', '방면/NNG', '방문/NNG', '방법/NNG', '방사능/NNG', '방사선/NNG', '방사성/NNG', '방송/NNG', '방송사/NNG', '방송통신위원회/NNG', '방식/NNG', '방안/NNG', '방지/NNG', '방청석/NNG', '방치/NNG', '방침/NNG', '방통위/NNG', '방해/NNG', '방향/NNG', '배/NNG', '배/NNP', '배경/NNG', '배려/NNG', '배분/NNG', '배상/NNG', '배수로/NNG', '배영/NNG', '배우/NNG', '배우자/NNG', '배재현/NNP', '배점/NNG', '배정/NNG', '배정원/NNP', '배제/NNG', '배차/NNG', '배치/NNG', '배치기준/NNG', '배터리/NNG', '배포/NNG', '백화점/NNG', '밴/NNG', '버스/NNG', '버스정류장/NNG', '버클리대학/NNP', '버튼/NNG', '번호/NNG', '벌금/NNG', '벌금형/NNG', '범위/NNG', '범인/NNG', '범정부/NNG', '범죄/NNG', '범죄자/NNG', '범주/NNG', '범행/NNG', '법/NNG', '법관/NNG', '법규/NNG', '법령/NNG', '법률/NNG', '법률’/NNG', '법률안/NNG', '법률적/NNG', '법률조력/NNG', '법률조력인/NNG', '법무법인/NNG', '법무부/NNG', '법안/NNG', '법안심사소위원회/NNG', '법원/NNG', '법인/NNG', '법적/NNG', '법적지위/NNG', '법정/NNG', '법정기준/NNG', '법정다툼/NNG', '법정대수/NNG', '법정정원/NNG', '법제처/NNG', '법조계/NNG', '베란다/NNG', '베이징/NNP', '베일리/NNP', '벽/NNG', '벽면/NNG', '변/NNG', '변/NNP', '변경/NNG', '변기/NNG', '변동/NNG', '변승일/NNG', '변용찬/NNP', '변이유전자/NNG', '변호사/NNG', '변화/NNG', '별/NNG', '별관/NNG', '별도/NNG', '병/NNG', '병가/NNG', '병상/NNG', '병실/NNG', '병원/NNG', '병율/NNG', '보/NNG', '보건/NNG', '보건복지/NNG', '보건복지부/NNG', '보건복지위원회/NNG', '보건복지인력개발/NNG', '보건소/NNG', '보고/NNG', '보고서/NNG', '보관/NNG', '보급/NNG', '보도/NNG', '보도자료/NNG', '보상/NNG', '보수/NNG', '보신각/NNP', '보완/NNG', '보완대체의사소통/NNG', '보유/NNG', '보육/NNG', '보육교사/NNG', '보육서비스/NNG', '보육시설/NNG', '보일러/NNG', '보장/NNG', '보장구/NNG', '보조/NNG', '보조공학/NNG', '보조공학기기/NNG', '보조금/NNG', '보조기구/NNG', '보조기기/NNG', '보조인/NNG', '보조인력/NNG', '보좌관/NNG', '보청기/NNG', '보치아/NNG', '보치아/NNP', '보통/NNG', '보통사람/NNG', '보편적/NNG', '보행/NNG', '보행자/NNG', '보험/NNG', '보험급여/NNG', '보험료/NNG', '보험료율/NNG', '보험적용/NNG', '보호/NNG', '보호관찰/NNG', '보호단체/NNG', '보호시설/NNG', '보호자/NNG', '보호작업장/NNG', '복권기금/NNG', '복도/NNG', '복사/NNG', '복수/NNG', '복식/NNG', '복지/NNG', '복지국가/NNG', '복지급/NNG', '복지대상자/NNG', '복지부/NNG', '복지부장관/NNG', '복지분야/NNG', '복지사업/NNG', '복지서비스/NNG', '복지시설/NNG', '복지언론/NNG', '복지연합신문/NNG', '복지예산/NNG', '복지정보연계시스템/NNG', '복지정책/NNG', '복지제도/NNG', '복지증진/NNG', '복지지원/NNG', '복합부위통증증후군/NNG', '복합적/NNG', '본격/NNG', '본격적/NNG', '본격화/NNG', '본대/NNG', '본부/NNG', '본선/NNG', '본인/NNG', '본인부담/NNG', '본인부담금/NNG', '본인부담상한제/NNG', '본인부담액/NNG', '본인증/NNG', '본회/NNG', '본회의/NNG', '볼라드/NNG', '봄/NNG', '봉사/NNG', '봉사활동/NNG', '부/NNG', '부가급여/NNG', '부과/NNG', '부과기준/NNG', '부과대상/NNG', '부근/NNG', '부담/NNG', '부담금/NNG', '부담기초액/NNG', '부당/NNG', '부당청구/NNG', '부대행사/NNG', '부동산/NNG', '부모/NNG', '부모님/NNG', '부문/NNG', '부부/NNG', '부분/NNG', '부산/NNP', '부산시/NNP', '부상/NNG', '부서/NNG', '부설/NNG', '부스/NNG', '부실/NNG', '부양/NNG', '부양의무/NNG', '부양의무자/NNG', '부양의무제/NNG', '부양자/NNG', '부여/NNG', '부위/NNG', '부위원장/NNG', '부의장/NNG', '부인/NNG', '부자/NNG', '부장/NNG', '부장검사/NNG', '부장판사/NNG', '부재/NNG', '부재자신고/NNG', '부정/NNG', '부정수급/NNG', '부정입학/NNG', '부정적/NNG', '부정행위/NNG', '부족/NNG', '부지/NNG', '부착/NNG', '부처/NNG', '부천시/NNP', '부탁/NNG', '부터/NNG', '부패/NNG', '부활/NNG', '부회장/NNG', '북/NNP', '북부/NNG', '북한/NNP', '분/NNG', '분과/NNG', '분기/NNG', '분노/NNG', '분당/NNP', '분리/NNG', '분석/NNG', '분석결과/NNG', '분야/NNG', '분양/NNG', '분열/NNG', '분위기/NNG', '분포/NNG', '불/NNG', '불가/NNG', '불가능/NNG', '불구/NNG', '불구속/NNG', '불균형/NNG', '불만/NNG', '불면증/NNG', '불법/NNG', '불법이용자/NNG', '불법적용/NNG', '불안/NNG', '불안감/NNG', '불용액/NNG', '불의/NNG', '불이익/NNG', '불편/NNG', '불편사항/NNG', '브라질/NNP', '브리핑/NNG', '블로그/NNG', '비/NNG', '비경인구/NNG', '비공인/NNG', '비교/NNG', '비교적/NNG', '비난/NNG', '비디오/NNG', '비례대표/NNG', '비리/NNG', '비밀/NNG', '비상호출버튼/NNG', '비상호출벨/NNG', '비옷/NNG', '비용/NNG', '비율/NNG', '비장애/NNG', '비장애인/NNG', '비장애인화장실/NNG', '비장애청소년/NNG', '비전/NNG', '비정규직/NNG', '비중/NNG', '비판/NNG', '비하/NNG', '비행기/NNG', '비효율/NNG', '빈곤/NNG', '빈곤선/NNG', '빈곤층/NNG', '빈소/NNG', '빌딩/NNG', '빗속/NNG', '빗줄기/NNG', '빚/NNG', '빛/NNG', '빨래/NNG', '빵/NNG', '사/NNG', '사각지대/NNG', '사거리/NNG', '사건/NNG', '사건’/NNG', '사격/NNG', '사격연맹/NNG', '사고/NNG', '사과/NNG', '사기/NNG', '사냥꾼/NNG', '사람/NNG', '사랑/NNG', '사랑아/NNG', '사랑아’/NNG', '사례/NNG', '사례관리/NNG', '사립학교/NNG', '사망/NNG', '사망자/NNG', '사명감/NNG', '사무/NNG', '사무관/NNG', '사무국/NNG', '사무실/NNG', '사무차장/NNG', '사무처/NNG', '사무처장/NNG', '사무총장/NNG', '사범/NNG', '사상/NNG', '사생활/NNG', '사설/NNG', '사실/NNG', '사실상/NNG', '사안/NNG', '사업/NNG', '사업’/NNG', '사업계획/NNG', '사업계획서/NNG', '사업비/NNG', '사업소/NNG', '사업자/NNG', '사업장/NNG', '사업주/NNG', '사업체/NNG', '사연/NNG', '사용/NNG', '사용자/NNG', '사원/NNG', '사위/NNG', '사유/NNG', '사이/NNG', '사이클/NNG', '사이트/NNG', '사인/NNG', '사장/NNG', '사전/NNG', '사정/NNG', '사진/NNG', '사태/NNG', '사퇴/NNG', '사항/NNG', '사회/NNG', '사회공헌활동/NNG', '사회구성원/NNG', '사회단체/NNG', '사회문제/NNG', '사회복귀/NNG', '사회복귀시설/NNG', '사회복지/NNG', '사회복지대상자/NNG', '사회복지법인/NNG', '사회복지사/NNG', '사회복지사업법/NNG', '사회복지서비스/NNG', '사회복지시설/NNG', '사회복지예산/NNG', '사회복지통합관리망/NNG', '사회복지학과/NNG', '사회봉사/NNG', '사회생활/NNG', '사회서비스/NNG', '사회성/NNG', '사회안전망/NNG', '사회적/NNG', '사회적기업/NNG', '사회참여/NNG', '사회통합/NNG', '사회학과/NNG', '사회활동/NNG', '사후/NNG', '사후관리/NNG', '사후환급/NNG', '삭감/NNG', '삭발/NNG', '삭제/NNG', '산/NNG', '산간/NNG', '산모/NNG', '산업/NNG', '산재장애인/NNG', '산정/NNG', '산하/NNG', '산하기관/NNG', '살림/NNG', '살인/NNG', '삶/NNG', '삼성코닝정밀소재/NNP', '상/NNG', '상고/NNG', '상공회의소/NNG', '상관관계/NNG', '상근/NNG', '상금/NNG', '상단/NNG', '상담/NNG', '상담소/NNG', '상담원/NNG', '상당/NNG', '상당수/NNG', '상대/NNG', '상대적/NNG', '상도동/NNP', '상반기/NNG', '상생/NNG', '상설/NNG', '상습/NNG', '상습적/NNG', '상승/NNG', '상시/NNG', '상시근로자/NNG', '상시근로자수/NNG', '상식/NNG', '상실/NNG', '상여금/NNG', '상영/NNG', '상원/NNG', '상위/NNG', '상임공동대표/NNG', '상임대표/NNG', '상임위/NNG', '상처/NNG', '상태/NNG', '상태”/NNG', '상품/NNG', '상하/NNG', '상한/NNG', '상한선/NNG', '상향/NNG', '상호/NNG', '상호출버튼/NNG', '상호출벨/NNG', '상황/NNG', '새/NNG', '새누리당/NNG', '새누리당/NNP', '새벽/NNG', '새해/NNG', '색깔/NNG', '색상/NNG', '샘/NNG', '생/NNG', '생각/NNG', '생계/NNG', '생계급여/NNG', '생명/NNG', '생산/NNG', '생산성/NNG', '생산시설/NNG', '생산적/NNG', '생산품/NNG', '생성/NNG', '생애주기/NNG', '생존/NNG', '생존권/NNG', '생활/NNG', '생활보장/NNG', '생활비/NNG', '생활수준/NNG', '생활시설/NNG', '생활안정/NNG', '생활체육/NNG', '생활환경/NNG', '샤워/NNG', '서/NNP', '서구/NNP', '서로/NNG', '서류/NNG', '서류전형/NNG', '서명/NNG', '서명운동/NNG', '서민/NNG', '서비스/NNG', '서비스센터/NNG', '서비스제공/NNG', '서울/NNP', '서울대/NNP', '서울맹학교/NNP', '서울메트로/NNP', '서울시/NNP', '서울중앙/NNP', '서인환/NNP', '서초구/NNP', '서초센터/NNP', '서초장애인자립생활센터/NNG', '선/NNG', '선거/NNG', '선거공보/NNG', '선거과정/NNG', '선거방송/NNG', '선거사무소/NNG', '선거인/NNG', '선고/NNG', '선관위/NNG', '선물/NNG', '선발/NNG', '선배/NNG', '선별적/NNG', '선생/NNG', '선생님/NNG', '선수/NNG', '선수단/NNG', '선수단장/NNG', '선순환/NNG', '선언/NNG', '선언문/NNG', '선의/NNG', '선정/NNG', '선진국/NNG', '선진통일당/NNP', '선천적/NNG', '선출/NNG', '선택/NNG', '선택권/NNG', '선포/NNG', '설계/NNG', '설립/NNG', '설립요건/NNG', '설명/NNG', '설명회’/NNG', '설문조사/NNG', '설비/NNG', '설정/NNG', '설치/NNG', '설치기준/NNG', '설치율/NNG', '섭취/NNG', '성/NNG', '성/NNP', '성격/NNG', '성공/NNG', '성공기원/NNG', '성공적/NNG', '성과/NNG', '성관계/NNG', '성교육/NNG', '성균관대/NNP', '성남시/NNP', '성년/NNP', '성능/NNG', '성매매/NNG', '성명/NNG', '성명서/NNG', '성범죄/NNG', '성범죄경력/NNG', '성범죄자/NNG', '성별/NNG', '성보호/NNG', '성북구/NNP', '성인/NNG', '성장/NNG', '성장애/NNG', '성장애인/NNG', '성적/NNG', '성추행/NNG', '성취/NNG', '성폭력/NNG', '성폭력범죄/NNG', '성폭행/NNG', '성향/NNG', '성희롱/NNG', '세계/NNG', '세계대회/NNG', '세계랭킹/NNG', '세계보건기구/NNG', '세계신기록/NNG', '세계장애대회/NNG', '세계장애인/NNG', '세계적/NNG', '세금/NNG', '세기/NNG', '세기보청기/NNG', '세대/NNG', '세레모니/NNG', '세력/NNG', '세면대/NNG', '세미나/NNG', '세부/NNG', '세부기준/NNG', '세부적/NNG', '세분화/NNG', '세상/NNG', '세상’/NNG', '세상걷기/NNG', '세션/NNG', '세종/NNP', '세종시/NNP', '세트/NNG', '센서/NNG', '센터/NNG', '소/NNG', '소감/NNG', '소개/NNG', '소견/NNG', '소관/NNG', '소규모/NNG', '소극적/NNG', '소년/NNG', '소득/NNG', '소득공제/NNG', '소득기준/NNG', '소득보장/NNG', '소득수준/NNG', '소득월액/NNG', '소득월액보험료/NNG', '소득인정액/NNG', '소득하위/NNG', '소리/NNG', '소망/NNG', '소명/NNG', '소방서/NNG', '소변기/NNG', '소비자/NNG', '소속/NNG', '소송/NNG', '소수자/NNG', '소식/NNG', '소식지/NNG', '소아/NNG', '소아마비/NNG', '소외/NNG', '소외계층/NNG', '소장/NNG', '소재/NNG', '소재지/NNG', '소지/NNG', '소총/NNG', '소통/NNG', '소형/NNG', '속/NNG', '속도/NNG', '손/NNG', '손가락/NNG', '손길/NNG', '손목/NNG', '손발/NNG', '손병준/NNP', '손상/NNG', '손잡이/NNG', '손학규/NNP', '손해배상/NNG', '송/NNP', '송도/NNP', '송도컨벤시아/NNP', '송중기/NNP', '송출/NNG', '쇄신/NNG', '쇼핑/NNG', '수/NNG', '수/NNP', '수가/NNG', '수검률/NNG', '수급/NNG', '수급권/NNG', '수급권자/NNG', '수급비/NNG', '수급이력/NNG', '수급자/NNG', '수급자격/NNG', '수년/NNG', '수능/NNG', '수단/NNG', '수당/NNG', '수도/NNG', '수도권/NNG', '수동/NNG', '수동휠체어/NNG', '수렴/NNG', '수리/NNG', '수립/NNG', '수면/NNG', '수면무호흡/NNG', '수모/NNG', '수발/NNG', '수법/NNG', '수분/NNG', '수사/NNG', '수사기관/NNG', '수상/NNG', '수상돌기/NNG', '수상자/NNG', '수수마/NNG', '수술/NNG', '수술실/NNG', '수신/NNG', '수양/NNG', '수업/NNG', '수여/NNG', '수여식/NNG', '수영/NNG', '수영장/NNG', '수요/NNG', '수요자/NNG', '수요조사/NNG', '수용/NNG', '수용여부/NNG', '수원/NNP', '수원시/NNP', '수원역/NNP', '수의계약/NNG', '수익/NNG', '수익성/NNG', '수입/NNG', '수정/NNG', '수족구병/NNG', '수준/NNG', '수준”/NNG', '수직형리프트/NNG', '수질/NNG', '수차례/NNG', '수출입은행/NNG', '수치/NNG', '수평/NNG', '수학/NNG', '수행/NNG', '수행기관/NNG', '수험생/NNG', '수혜자/NNG', '수화/NNG', '수화언어/NNG', '수화언어기본법/NNG', '수화통역/NNG', '수화통역사/NNG', '수화통역센터/NNG', '숙박시설/NNG', '숙소/NNG', '숙자매/NNG', '순/NNG', '순간/NNG', '순서/NNG', '순위/NNG', '순차적/NNG', '술/NNG', '숨/NNG', '숫자/NNG', '쉘터/NNG', '스/NNG', '스마트교실/NNG', '스마트기기/NNG', '스마트러닝/NNG', '스마트폰/NNG', '스스로/NNG', '스웨덴/NNP', '스코어/NNG', '스쿠터/NNG', '스쿱/NNG', '스크린/NNG', '스크린도어/NNP', '스타/NNG', '스타디움/NNG', '스타벅스/NNP', '스탠딩/NNG', '스텝/NNG', '스텝차량/NNG', '스튜디오/NNG', '스트레스/NNG', '스틸컷/NNG', '스페셜/NNG', '스페셜올림픽/NNG', '스페인/NNP', '스포츠/NNG', '슬로건/NNG', '승/NNG', '승강기/NNG', '승강장/NNG', '승객/NNG', '승리/NNG', '승부/NNG', '승용차/NNG', '승인/NNG', '시/NNG', '시가/NNG', '시각/NNG', '시각장애/NNG', '시각장애인/NNG', '시각장애인복지관/NNG', '시각장애인아동학교/NNG', '시각장애인연합/NNG', '시각장애인연합회/NNG', '시각장애인연합회장/NNG', '시각장애인용/NNG', '시간/NNG', '시간동안/NNG', '시교육청/NNG', '시군/NNG', '시급/NNG', '시기/NNG', '시내/NNG', '시내버스/NNG', '시대/NNG', '시대적/NNG', '시도/NNG', '시도교육청/NNG', '시라이시/NNP', '시력/NNG', '시민/NNG', '시민단체/NNG', '시민복지기준/NNG', '시민사회/NNG', '시민사회단체/NNG', '시민인권보호관/NNG', '시범/NNG', '시범사업/NNG', '시범운영/NNG', '시비/NNG', '시상/NNG', '시상대/NNG', '시상식/NNG', '시선/NNG', '시설/NNG', '시설물/NNG', '시설운영/NNG', '시설이용/NNG', '시설입소/NNG', '시설장/NNG', '시설장애인/NNG', '시설주/NNG', '시속/NNG', '시스템/NNG', '시위/NNG', '시의회/NNG', '시일/NNG', '시작/NNG', '시장/NNG', '시절/NNG', '시점/NNG', '시정/NNG', '시정명령/NNG', '시중/NNG', '시즌/NNG', '시청/NNG', '시청자/NNG', '시카고/NNP', '시티투어버스/NNP', '시행/NNG', '시행규칙/NNG', '시행령/NNG', '시험/NNG', '시혜/NNG', '시혜적/NNG', '식당/NNG', '식물원/NNG', '식사/NNG', '식약청/NNG', '식중독/NNG', '식품/NNG', '식품위생법/NNG', '신/NNG', '신/NNP', '신경/NNG', '신경전달물질/NNG', '신고/NNG', '신고센터/NNG', '신고자/NNG', '신규/NNG', '신규도입/NNG', '신기록/NNG', '신뢰/NNG', '신뢰관계/NNG', '신뢰관계인/NNG', '신문/NNG', '신부/NNG', '신분/NNG', '신분증/NNG', '신빙성/NNG', '신상공개/NNG', '신상정보/NNG', '신생아/NNG', '신설/NNG', '신용카드/NNG', '신장/NNG', '신청/NNG', '신청서/NNG', '신청인/NNG', '신청일/NNG', '신청자/NNG', '신청자격/NNG', '신체/NNG', '신체적/NNG', '신호/NNG', '신혼부부/NNG', '신혼여행/NNG', '실내/NNG', '실력/NNG', '실망/NNG', '실무자/NNG', '실수/NNG', '실습/NNG', '실시/NNG', '실시간/NNG', '실업급여/NNG', '실외/NNG', '실장/NNG', '실적/NNG', '실정/NNG', '실정”/NNG', '실제/NNG', '실제소득/NNG', '실질/NNG', '실질적/NNG', '실천/NNG', '실체적/NNG', '실태/NNG', '실태조사/NNG', '실패/NNG', '실행/NNG', '실험/NNG', '실현/NNG', '실효/NNG', '실효성/NNG', '심/NNG', '심/NNP', '심각/NNG', '심각성/NNG', '심도/NNG', '심리/NNG', '심리적/NNG', '심리치료/NNG', '심사/NNG', '심상정/NNP', '심의/NNG', '심장/NNG', '심재용/NNP', '심층/NNG', '심포지엄/NNG', '싸움/NNG', '쌍/NNG', '쌍둥이/NNG', '아기/NNG', '아나운서/NNG', '아내/NNG', '아동/NNG', '아동보호구역/NNG', '아동복지/NNG', '아동복지시설/NNG', '아들/NNG', '아래/NNG', '아레나/NNP', '아르바이트/NNG', '아메리칸/NNG', '아무것/NNG', '아밀로이드/NNP', '아버지/NNG', '아쉬움/NNG', '아시아/NNP', '아시안게임/NNG', '아웃/NNG', '아웃소싱/NNG', '아이/NNG', '아이돌보미/NNG', '아이돌봄/NNG', '아이돌봄서비스/NNG', '아이디어/NNG', '아이봇/NNG', '아이패드/NNG', '아침/NNG', '아카데미/NNG', '아태/NNP', '아태장애/NNG', '아태장애인/NNG', '아태장애인대회/NNG', '아태장애인연합/NNG', '아태장애포럼/NNG', '아트홀/NNG', '아파트/NNG', '아프리카/NNP', '아픔/NNG', '악몽/NNG', '안/NNG', '안/NNP', '안건/NNG', '안광훈/NNP', '안내/NNG', '안내견/NNG', '안내문/NNG', '안내서비스/NNG', '안내판/NNG', '안랩/NNG', '안마/NNG', '안마사/NNG', '안마사자격제도/NNG', '안마사협회/NNG', '안보/NNG', '안산/NNP', '안산시/NNP', '안산시장/NNG', '안상수/NNP', '안양시/NNP', '안전/NNG', '안전망/NNG', '안전사고/NNG', '안전상비의약품/NNG', '안전장치/NNG', '안정/NNG', '안정적/NNG', '안진환/NNP', '안쪽/NNG', '안철수/NNP', '안팎/NNG', '알츠하이머/NNG', '알코올/NNG', '알코올성/NNG', '암/NNG', '압도적/NNG', '압수수색/NNG', '앞/NNG', '앞자리/NNG', '애/NNG', '애플리케이션/NNG', '액셀/NNG', '액수/NNG', '야/NNG', '야간/NNG', '야권/NNG', '야기/NNG', '야당/NNG', '야외/NNG', '야학/NNG', '약/NNG', '약간/NNG', '약물/NNG', '약물치료/NNG', '약속/NNG', '약자/NNG', '양/NNG', '양/NNP', '양궁/NNG', '양극화/NNG', '양도/NNG', '양상/NNG', '양성/NNG', '양시체육관/NNG', '양양/NNP', '양양군/NNP', '양양군청/NNG', '양양군청/NNP', '양옆/NNG', '양육/NNG', '양육보조금/NNG', '양육수당/NNG', '양육지원/NNG', '양자/NNG', '양적/NNG', '양질/NNG', '양쪽/NNG', '양형기준/NNG', '양호/NNP', '얘기/NNG', '어깨/NNG', '어려움/NNG', '어르신/NNG', '어른/NNG', '어린아이/NNG', '어린이/NNG', '어린이집/NNG', '어머니/NNG', '어제/NNG', '어플리케이션/NNG', '억제/NNG', '언급/NNG', '언론/NNG', '언어/NNG', '언어선택권/NNG', '언어장애/NNG', '언어장애인/NNG', '언어재활/NNG', '언어재활사/NNG', '언어치료/NNG', '얼굴/NNG', '얼마/NNG', '엄마/NNG', '업무/NNG', '업무과중/NNG', '업무보고/NNG', '업무처리지원/NNG', '업무협약/NNG', '업소/NNG', '업종/NNG', '업주/NNG', '업체/NNG', '엉망/NNG', '에너지/NNG', '에빌리/NNP', '에세이/NNG', '에스캅/NNG', '에스컬레이터/NNG', '에스코트/NNG', '에어컨/NNG', '에이블뉴스/NNG', '에이블뉴스/NNP', '에이블뉴스제휴사/NNP', '에이블복지재단/NNG', '에이블복지재단/NNP', '에페/NNG', '에픽/NNG', '엑셀/NNG', '엑스포/NNG', '엘리베이터/NNG', '여/NNG', '여가/NNG', '여가부/NNG', '여건/NNG', '여기저기/NNG', '여닫이/NNG', '여닫이문/NNG', '여력/NNG', '여론/NNG', '여름/NNG', '여름철/NNG', '여부/NNG', '여성/NNG', '여성가족부/NNG', '여성부/NNG', '여성장애인/NNG', '여성장애인기본법/NNG', '여성장애인대회/NNG', '여성장애인연합/NNG', '여성장애인화장실/NNG', '여아/NNG', '여야/NNG', '여의도/NNP', '여의도공원/NNP', '여자/NNG', '여자선수/NNG', '여장연/NNG', '여정/NNG', '여중생/NNG', '여학생/NNG', '여행/NNG', '여행바우처/NNG', '역/NNG', '역대/NNG', '역도/NNG', '역량/NNG', '역량강화/NNG', '역사/NNG', '역점/NNG', '역할/NNG', '연/NNG', '연간/NNG', '연결/NNG', '연계/NNG', '연계고용/NNG', '연관/NNG', '연구/NNG', '연구개발/NNG', '연구결과/NNG', '연구결과’/NNG', '연구기관/NNG', '연구소/NNG', '연구용역/NNG', '연구원/NNG', '연구위원/NNG', '연구윤리헌장/NNG', '연구팀/NNG', '연금/NNG', '연금공단/NNG', '연금보험료/NNG', '연기/NNG', '연내/NNG', '연대/NNG', '연대회의/NNG', '연도/NNG', '연락/NNG', '연령/NNG', '연령층/NNG', '연료/NNG', '연료비/NNG', '연말/NNG', '연말정산/NNG', '연맹/NNG', '연면적/NNG', '연명치료/NNG', '연방/NNG', '연속/NNG', '연수/NNG', '연일/NNG', '연임/NNG', '연장/NNG', '연탄/NNG', '연패/NNG', '연평균/NNG', '연합뉴스/NNP', '연합회/NNG', '열기/NNG', '열망/NNG', '열전/NNG', '열정/NNG', '열차/NNG', '염/NNP', '염두/NNG', '염려/NNG', '염색체/NNG', '염원/NNG', '엽서/NNG', '영/NNP', '영광/NNG', '영국/NNP', '영남/NNP', '영도구/NNP', '영동/NNP', '영리/NNG', '영상/NNG', '영어/NNP', '영업/NNG', '영역/NNG', '영예/NNG', '영유아/NNG', '영향/NNG', '영향력/NNG', '영혼/NNG', '영화/NNG', '영화관/NNG', '영화관람/NNG', '영화제/NNG', '옆/NNG', '예/NNG', '예결위/NNG', '예고/NNG', '예방/NNG', '예방교육/NNG', '예방접종/NNG', '예비/NNG', '예비인증/NNG', '예비후보/NNG', '예비후보자/NNG', '예산/NNG', '예산결산특별위원회/NNG', '예산안/NNG', '예산액/NNG', '예산제도/NNG', '예산편성/NNG', '예상/NNG', '예선/NNG', '예술/NNG', '예술가/NNG', '예술인/NNG', '예술적/NNG', '예약/NNG', '예외/NNG', '예의/NNG', '예전/NNG', '예정/NNG', '예정”/NNG', '오/NNP', '오늘/NNG', '오동/NNG', '오류/NNG', '오른쪽/NNG', '오마이뉴스/NNP', '오븐/NNG', '오스틴/NNP', '오염/NNG', '오전/NNG', '오제세/NNP', '오토모토/NNG', '오픈/NNG', '오해/NNG', '오후/NNG', '옥외/NNG', '옥천/NNP', '옥천군청/NNP', '옥천센터/NNP', '온라인/NNG', '온몸/NNG', '올/NNG', '올림픽/NNG', '올해/NNG', '옷/NNG', '옷장/NNG', '완속/NNG', '완전/NNG', '완화/NNG', '왜곡/NNG', '외교/NNG', '외국/NNG', '외국어/NNG', '외국인/NNG', '외면/NNG', '외부/NNG', '외상/NNG', '외적/NNG', '외출/NNG', '외침/NNG', '왼쪽/NNG', '요건/NNG', '요구/NNG', '요구공약/NNG', '요구사항/NNG', '요구서/NNG', '요구안/NNG', '요금/NNG', '요금감면/NNG', '요소/NNG', '요양기관/NNG', '요양병원/NNG', '요양시설/NNG', '요양원/NNG', '요원/NNG', '요인/NNG', '요즘/NNG', '요청/NNG', '욕구/NNG', '용기/NNG', '용도/NNG', '용변/NNG', '용산구/NNP', '용어/NNG', '용역/NNG', '용역서비스/NNG', '용의자/NNG', '용인시/NNP', '우려/NNG', '우리나라/NNG', '우산/NNG', '우선/NNG', '우선구매/NNG', '우선구매제도/NNG', '우선순위/NNG', '우선적/NNG', '우수/NNG', '우수사례/NNG', '우수상/NNG', '우승/NNG', '우울증/NNG', '우의/NNG', '우측/NNG', '우편/NNG', '운동/NNG', '운동장/NNG', '운동화/NNG', '운영/NNG', '운영기준/NNG', '운영방안/NNG', '운영비/NNG', '운영위원/NNG', '운영위원회/NNG', '운용/NNG', '운전/NNG', '운전면허시험장/NNG', '운전자/NNG', '운행/NNG', '운행지역/NNG', '울산/NNG', '울산/NNP', '울산과기대/NNP', '울산시/NNP', '움직임/NNG', '웃음/NNG', '워싱턴/NNP', '워크숍/NNG', '워크투게더센터/NNG', '원거리/NNG', '원격교육지원/NNG', '원고/NNG', '원래/NNG', '원생/NNG', '원서/NNG', '원숭이/NNG', '원스탑/NNG', '원심/NNG', '원안/NNG', '원인/NNG', '원장/NNG', '원전/NNG', '원천/NNG', '원칙/NNG', '원칙적/NNG', '원탁회의/NNG', '월/NNG', '월평균/NNG', '월평균소득/NNG', '웹/NNG', '웹사이트/NNG', '웹앤미디어/NNG', '웹접근성/NNG', '위/NNG', '위기/NNG', '위력/NNG', '위령제/NNG', '위반/NNG', '위법/NNG', '위상/NNG', '위안부/NNG', '위원/NNG', '위원장/NNG', '위원회/NNG', '위주/NNG', '위촉/NNG', '위축/NNG', '위치/NNG', '위캔/NNG', '위탁/NNG', '위헌법률심판/NNG', '위험/NNG', '위험성/NNG', '위협/NNG', '윌/NNP', '유/NNP', '유가족/NNG', '유관기관/NNG', '유권자/NNG', '유급화/NNG', '유기적/NNG', '유니버설/NNP', '유도/NNG', '유럽/NNP', '유리/NNG', '유린/NNG', '유명/NNG', '유모차/NNG', '유무/NNG', '유병훈/NNP', '유사/NNG', '유승희/NNP', '유시/NNG', '유아/NNG', '유아특수교사/NNG', '유아특수교육/NNG', '유아학비/NNG', '유엔/NNP', '유엔에스캅/NNP', '유연고용/NNG', '유은혜/NNP', '유전자/NNG', '유전자검사/NNG', '유지/NNG', '유지호/NNP', '유치원/NNG', '유치장/NNG', '유튜브/NNP', '유형/NNG', '유효기간/NNG', '육교/NNG', '육상/NNG', '육성/NNG', '육성법/NNG', '육아/NNG', '육체/NNG', '윤/NNP', '윤석용/NNP', '윤선아/NNP', '윤식/NNG', '윤종술/NNP', '율촌/NNP', '융합과학기술대학원장/NNG', '은/NNG', '은메달/NNG', '은지/NNP', '은행/NNG', '음/NNG', '음란물/NNG', '음성/NNG', '음성언어/NNG', '음성유도기/NNG', '음식/NNG', '음식점/NNG', '음악/NNG', '음악회/NNG', '음주/NNG', '음향신호기/NNG', '읍/NNG', '읍면동/NNG', '읍사무소/NNG', '응급의료/NNG', '응답/NNG', '응답자/NNG', '응시/NNG', '응시원서/NNG', '응시자/NNG', '응원/NNG', '의견/NNG', '의견수렴/NNG', '의결/NNG', '의구심/NNG', '의대/NNG', '의도/NNG', '의뢰/NNG', '의료/NNG', '의료급여/NNG', '의료기관/NNG', '의료법/NNG', '의료비/NNG', '의료서비스/NNG', '의료적/NNG', '의료진/NNG', '의무/NNG', '의무건설/NNG', '의무고용/NNG', '의무고용률/NNG', '의무교육/NNG', '의무대수/NNG', '의무적/NNG', '의무화/NNG', '의문/NNG', '의미/NNG', '의사/NNG', '의사소통/NNG', '의사표현/NNG', '의수/NNG', '의식/NNG', '의심/NNG', '의원/NNG', '의원실/NNG', '의원회관/NNG', '의자/NNG', '의장/NNG', '의정부/NNP', '의정부경전철/NNG', '의정부시/NNP', '의제/NNG', '의족/NNG', '의존증/NNG', '의지/NNG', '의학적/NNG', '의혹/NNG', '의회/NNG', '이/NNG', '이/NNP', '이강천/NNP', '이견/NNG', '이경환/NNP', '이광동/NNP', '이광원/NNP', '이권희/NNP', '이글/NNP', '이낙연/NNP', '이날/NNG', '이내/NNG', '이념/NNG', '이달/NNG', '이동/NNG', '이동빨래방/NNG', '이동전화/NNG', '이동통신/NNG', '이동편/NNG', '이동편의시설/NNG', '이동편의증진법/NNG', '이때/NNG', '이력/NNG', '이력서/NNG', '이룸센터/NNG', '이룸홀/NNP', '이름/NNG', '이마트/NNP', '이메일/NNG', '이명박/NNP', '이목희/NNP', '이미용/NNG', '이미지/NNG', '이번/NNG', '이벤트/NNG', '이복/NNP', '이복남/NNP', '이불/NNG', '이사/NNG', '이사장/NNG', '이사회/NNG', '이상/NNG', '이상’/NNG', '이상철/NNP', '이상현/NNP', '이상호/NNP', '이성규/NNP', '이송/NNG', '이수/NNG', '이슈/NNG', '이스라엘/NNP', '이승기/NNP', '이식/NNG', '이신형/NNG', '이야기/NNG', '이외/NNG', '이용/NNG', '이용권/NNP', '이용대상/NNG', '이용료/NNG', '이용률/NNG', '이용시간/NNG', '이용자/NNG', '이웃/NNG', '이유/NNG', '이윤리/NNP', '이의제기/NNG', '이익/NNG', '이인국/NNP', '이전/NNG', '이정선/NNP', '이정현/NNP', '이정희/NNP', '이제/NNG', '이주희/NNP', '이중/NNG', '이진섭/NNP', '이채필/NNP', '이탈리아/NNP', '이태승/NNP', '이틀/NNG', '이하/NNG', '이학영/NNP', '이해/NNG', '이행/NNG', '이행율/NNG', '이화숙/NNP', '이화여대/NNP', '이화여자대학교/NNP', '이후/NNG', '인/NNG', '인가구/NNG', '인간/NNG', '인건비/NNG', '인구/NNG', '인권/NNG', '인권단체/NNG', '인권보장/NNG', '인권보호기구/NNG', '인권센터/NNG', '인권위/NNG', '인권유린/NNG', '인권증진/NNG', '인권침해/NNG', '인근/NNG', '인기/NNG', '인도/NNG', '인도/NNP', '인디언/NNP', '인력/NNG', '인물/NNG', '인사/NNG', '인사청문회/NNG', '인상/NNG', '인생/NNG', '인쇄/NNG', '인시위/NNG', '인식/NNG', '인식개선/NNG', '인애학교/NNG', '인연/NNG', '인원/NNG', '인재/NNG', '인적사항/NNG', '인정/NNG', '인정점수/NNG', '인정조사표/NNG', '인증/NNG', '인증기준/NNG', '인증기준점/NNG', '인증제/NNG', '인지/NNG', '인지도/NNG', '인천/NNP', '인천광역시/NNP', '인천기계공고/NNP', '인천시/NNP', '인체/NNG', '인터넷/NNG', '인터뷰/NNG', '인턴/NNG', '인프라/NNG', '인플루엔자/NNG', '인화학교/NNG', '일/NNG', '일/NNP', '일”/NNG', '일각/NNG', '일괄/NNG', '일괄적/NNG', '일대/NNG', '일례/NNG', '일률적/NNG', '일리노이주/NNP', '일명/NNG', '일반/NNG', '일반교사/NNG', '일반국민/NNG', '일반음식점/NNG', '일반인/NNG', '일반적/NNG', '일반학교/NNG', '일본/NNP', '일부/NNG', '일부개정령안/NNG', '일부개정법률안/NNG', '일산/NNP', '일상/NNG', '일상생활/NNG', '일선/NNG', '일시/NNG', '일시적/NNG', '일요일/NNG', '일원/NNG', '일원화/NNG', '일자/NNG', '일자리/NNG', '일자리사업/NNG', '일정/NNG', '일정기간/NNG', '일정부분/NNG', '일터/NNG', '일환/NNG', '임/NNP', '임금/NNG', '임기/NNG', '임대료/NNG', '임대비용/NNG', '임대아파트/NNG', '임대주택/NNG', '임대주택사업/NNG', '임명/NNG', '임명장/NNG', '임산부/NNG', '임시/NNG', '임시경/NNG', '임신/NNG', '임용/NNG', '임용고시/NNG', '임우근/NNP', '임원/NNG', '임직원/NNG', '임채민/NNG', '임채민/NNP', '임태희/NNP', '입/NNG', '입구/NNG', '입법/NNG', '입법예고/NNG', '입사/NNG', '입상자/NNG', '입성/NNG', '입소/NNG', '입소자/NNG', '입양/NNG', '입원/NNG', '입원환자/NNG', '입장/NNG', '입장권/NNG', '입주/NNG', '입주자/NNG', '입찰/NNG', '입학/NNG', '입학사정관/NNG', '입학전형/NNG', '자/NNG', '자’/NNG', '자격/NNG', '자격기준/NNG', '자격논란/NNG', '자격상실/NNG', '자격증/NNG', '자기결정/NNG', '자기소개/NNG', '자녀/NNG', '자녀교육비/NNG', '자녀양육/NNG', '자동/NNG', '자동차/NNG', '자락길/NNG', '자료/NNG', '자료협조/NNG', '자리/NNG', '자립/NNG', '자립생활/NNG', '자립생활가정/NNG', '자립생활기술훈련/NNG', '자립생활센터/NNG', '자립센터/NNG', '자막/NNG', '자막방송/NNG', '자매/NNG', '자문/NNG', '자문단/NNG', '자문단구성/NNG', '자발적/NNG', '자부담/NNG', '자살/NNG', '자선/NNG', '자세/NNG', '자식/NNG', '자신/NNG', '자신감/NNG', '자연/NNG', '자원/NNG', '자원봉사/NNG', '자원봉사자/NNG', '자유/NNG', '자유형/NNG', '자율성/NNG', '자율적/NNG', '자전거/NNG', '자조모임/NNG', '자존/NNG', '자진/NNG', '자질/NNG', '자체/NNG', '자치구/NNG', '자치단체/NNG', '자택/NNG', '자판기/NNG', '자폐/NNG', '자폐성/NNG', '자폐성장애/NNG', '자폐성장애인/NNG', '자폐증/NNG', '자회사/NNG', '자회사형/NNG', '작/NNG', '작가/NNG', '작년/NNG', '작동/NNG', '작성/NNG', '작성자/NNG', '작업/NNG', '작업장/NNG', '작품/NNG', '잔/NNG', '잔디/NNG', '잠/NNG', '잠금장치/NNG', '잠깐/NNG', '잠자리/NNG', '잠재력/NNG', '잠재적/NNG', '잠정/NNG', '잣대/NNG', '장/NNG', '장/NNP', '장관/NNG', '장관급/NNG', '장기/NNG', '장기적/NNG', '장난/NNG', '장난감/NNG', '장례/NNG', '장례식/NNG', '장례식장/NNG', '장르/NNG', '장막/NNG', '장면/NNG', '장벽/NNG', '장비/NNG', '장비구입/NNG', '장소/NNG', '장시간/NNG', '장애/NNG', '장애/NNP', '장애계/NNP', '장애관련/NNG', '장애단체/NNG', '장애대학/NNP', '장애대학생/NNP', '장애등급/NNG', '장애등록/NNG', '장애물/NNG', '장애범주/NNG', '장애법/NNG', '장애부모/NNG', '장애상태/NNG', '장애수당/NNG', '장애수용도/NNG', '장애아/NNG', '장애아동/NNG', '장애아동복지지원법/NNG', '장애아동수당/NNP', '장애아이/NNG', '장애여성/NNG', '장애여성공감/NNG', '장애여성네트워크/NNP', '장애연금/NNG', '장애영유아/NNP', '장애예술인/NNG', '장애와인권발바닥행동/NNP', '장애우권익문제연구소/NNG', '장애유아/NNG', '장애유아/NNP', '장애유형/NNG', '장애유형/NNP', '장애유형별/NNG', '장애유형별/NNP', '장애이해교육/NNG', '장애인/NNG', '장애인가구/NNG', '장애인가정/NNG', '장애인개발/NNG', '장애인거주시설/NNG', '장애인계/NNG', '장애인고/NNG', '장애인고용/NNG', '장애인고용공단/NNG', '장애인고용률/NNG', '장애인고용부담금/NNG', '장애인고용장려금/NNG', '장애인고용촉진/NNG', '장애인공무원/NNG', '장애인교육/NNG', '장애인구/NNG', '장애인국/NNG', '장애인권리보장법/NNG', '장애인권리보장법제정연대/NNP', '장애인권리위원/NNG', '장애인권리위원회/NNG', '장애인권리협약/NNG', '장애인권익지킴이/NNG', '장애인근로자/NNG', '장애인기업/NNG', '장애인기업종합지원센터/NNG', '장애인단체/NNG', '장애인단체장/NNG', '장애인단체총연맹/NNG', '장애인당사자/NNG', '장애인등급/NNG', '장애인등록/NNG', '장애인럭비/NNG', '장애인문화예술축제/NNG', '장애인보조기구/NNG', '장애인복지/NNG', '장애인복지관/NNG', '장애인복지법/NNG', '장애인복지법’/NNG', '장애인복지사업/NNG', '장애인복지시설/NNG', '장애인복지신문/NNG', '장애인복지정책/NNG', '장애인부모회/NNG', '장애인부부/NNG', '장애인생산품/NNG', '장애인생활시설/NNG', '장애인생활신문/NNG', '장애인석/NNG', '장애인선수/NNG', '장애인숙박시설/NNG', '장애인스포츠/NNG', '장애인시설/NNG', '장애인식개선/NNG', '장애인신문/NNG', '장애인실태조사/NNG', '장애인연금/NNG', '장애인연금법/NNG', '장애인연맹/NNG', '장애인영화제/NNG', '장애인예산/NNG', '장애인예술/NNG', '장애인예술회관/NNG', '장애인올림픽/NNG', '장애인위원장/NNG', '장애인위원회/NNG', '장애인유권자/NNG', '장애인의무고용률/NNG', '장애인인권영화제/NNG', '장애인인권포럼/NNG', '장애인일자리/NNG', '장애인자동차/NNG', '장애인자립생활/NNG', '장애인자립생활보장법/NNG', '장애인자립생활센터/NNG', '장애인자립생활센터협의회/NNG', '장애인재활협회/NNG', '장애인전용관/NNG', '장애인전용주차구역/NNG', '장애인전용주차장/NNG', '장애인전용화장실/NNG', '장애인전환서비스센터/NNG', '장애인정보문화누리/NNG', '장애인정책/NNG', '장애인정책조정위원회/NNG', '장애인정책종합계획/NNG', '장애인좌석/NNG', '장애인주거지원연대/NNG', '장애인주차장/NNG', '장애인직업재활시설/NNG', '장애인차량/NNG', '장애인차별금지/NNG', '장애인차별금지법/NNG', '장애인차별금지추진연대/NNG', '장애인차별철폐연대/NNG', '장애인체육/NNG', '장애인체육관/NNG', '장애인체육회/NNG', '장애인체전/NNG', '장애인콜택시/NNG', '장애인콜택시용/NNG', '장애인편/NNG', '장애인편의/NNG', '장애인편의시설/NNG', '장애인표준사업장/NNG', '장애인협회/NNG', '장애인화/NNG', '장애인화장/NNG', '장애인화장실/NNG', '장애인활동가/NNG', '장애인활동보조/NNG', '장애인활동지원/NNG', '장애인활동지원법/NNG', '장애인활동지원사업/NNG', '장애인활동지원서비스/NNG', '장애인활동지원제/NNG', '장애인활동지원제도/NNG', '장애인활동지원제도개선위원회/NNG', '장애정도/NNG', '장애청년/NNG', '장애청년드림팀/NNG', '장애청소년/NNG', '장애체험/NNG', '장애체험교육/NNG', '장애특성/NNG', '장애학생/NNG', '장전동/NNP', '장점/NNG', '장차법/NNG', '장착/NNG', '장총/NNG', '장총/NNP', '장총련/NNP', '장춘배/NNP', '장치/NNG', '장특법/NNG', '장향숙/NNP', '재/NNG', '재건축/NNG', '재검토/NNG', '재난/NNG', '재능/NNG', '재단/NNG', '재발/NNG', '재발방지/NNG', '재벌/NNG', '재범/NNG', '재보궐/NNG', '재산/NNG', '재산가/NNG', '재산세/NNG', '재신청/NNG', '재심/NNG', '재원/NNG', '재인증/NNG', '재정/NNG', '재정상황/NNG', '재정여건/NNG', '재정자립도/NNG', '재정적/NNG', '재조사/NNG', '재질/NNG', '재판/NNG', '재판과정/NNG', '재판부/NNG', '재판정/NNG', '재판진행/NNG', '재학/NNG', '재학생/NNG', '재해/NNG', '재활/NNG', '재활병원/NNG', '재활보조기구/NNG', '재활서비스/NNG', '재활센터/NNG', '재활치료/NNG', '재활협회/NNG', '쟁점/NNG', '저녁/NNG', '저상버스/NNG', '저소득/NNG', '저소득층/NNG', '저시력/NNG', '저임금/NNG', '저작권자/NNG', '저작물/NNG', '저하/NNG', '적극/NNG', '적극적/NNG', '적성/NNG', '적용/NNG', '적용대상/NNG', '적용제외/NNG', '적응/NNG', '적정/NNG', '적정기준/NNG', '적정성/NNG', '적채현상/NNG', '적합/NNG', '전/NNG', '전/NNP', '전격/NNG', '전경/NNG', '전공/NNG', '전광우/NNP', '전국/NNG', '전국장애인부모연대/NNP', '전국적/NNG', '전국특수교육과대학생연합회/NNP', '전국특수교육과협의회/NNP', '전근배/NNP', '전기/NNG', '전기차/NNG', '전날/NNG', '전남/NNP', '전년/NNG', '전년대비/NNG', '전달/NNG', '전달체계/NNG', '전담/NNG', '전담부서/NNG', '전담인력/NNG', '전동/NNG', '전동보장구/NNG', '전동스쿠터/NNG', '전동휠체어/NNG', '전라남도/NNP', '전략/NNG', '전략적/NNG', '전력/NNG', '전망/NNG', '전면/NNG', '전명훈/NNP', '전문/NNG', '전문가/NNG', '전문기관/NNG', '전문병원/NNG', '전문성/NNG', '전문의/NNG', '전문인력/NNG', '전문적/NNG', '전민재/NNG', '전민재/NNP', '전반/NNG', '전반적/NNG', '전부/NNG', '전북/NNP', '전산/NNG', '전수조사/NNG', '전시/NNG', '전시회/NNG', '전신/NNG', '전액/NNG', '전업주부/NNG', '전역/NNG', '전염병/NNG', '전용/NNG', '전원/NNG', '전자/NNG', '전자발찌/NNG', '전자파/NNG', '전장연/NNP', '전쟁/NNG', '전적/NNG', '전제/NNG', '전주/NNP', '전체/NNG', '전체관람/NNG', '전체예산/NNG', '전체적/NNG', '전체회의/NNG', '전향적/NNG', '전형/NNG', '전형적/NNG', '전화/NNG', '전환/NNG', '절감/NNG', '절단/NNG', '절대적/NNG', '절망/NNG', '절반/NNG', '절벽/NNG', '절차/NNG', '점/NNG', '점거/NNG', '점검/NNG', '점검주기/NNG', '점수/NNG', '점심/NNG', '점심식사/NNG', '점자/NNG', '점자블록/NNG', '점자안내판/NNG', '점자표지판/NNG', '점자형/NNG', '접근/NNG', '접근권/NNG', '접근성/NNG', '접근성’/NNG', '접수/NNG', '접이식/NNG', '접전/NNG', '접촉/NNG', '정/NNP', '정권/NNG', '정권교체/NNG', '정규직/NNG', '정규직종/NNG', '정기/NNG', '정기적/NNG', '정당/NNG', '정덕/NNP', '정도/NNG', '정동호/NNP', '정론관/NNG', '정리/NNG', '정립회관/NNG', '정문/NNG', '정보/NNG', '정보접근/NNG', '정보제공/NNG', '정보통신/NNG', '정보통신망/NNG', '정보통신보조기기/NNG', '정부/NNG', '정부간고위급회의/NNG', '정부기관/NNG', '정부부처/NNG', '정부안/NNG', '정부예산안/NNG', '정부중앙청사/NNG', '정부차원/NNG', '정비/NNG', '정산/NNG', '정상/NNG', '정상숙/NNP', '정상인/NNG', '정상적/NNG', '정상철/NNP', '정상화/NNG', '정서/NNG', '정서적/NNG', '정성/NNG', '정세균/NNP', '정소영/NNP', '정승락/NNP', '정식/NNG', '정신/NNG', '정신건강/NNG', '정신보건법/NNG', '정신장애/NNG', '정신장애인/NNG', '정신적/NNG', '정신지체/NNG', '정신질환/NNG', '정신질환자/NNG', '정원/NNG', '정의/NNG', '정작/NNG', '정착/NNG', '정책/NNG', '정책기구/NNG', '정책솔루션/NNG', '정책실장/NNG', '정책위원장/NNG', '정책자문/NNG', '정책적/NNG', '정책토론회/NNG', '정책협약/NNG', '정치/NNG', '정치권/NNG', '정치적/NNG', '정호원/NNP', '정황/NNG', '제거/NNG', '제고/NNG', '제공/NNG', '제공기관/NNG', '제기/NNG', '제네바/NNP', '제도/NNG', '제도’/NNG', '제도개선/NNG', '제도적/NNG', '제도화/NNG', '제목/NNG', '제보/NNG', '제본/NNG', '제시/NNG', '제안/NNG', '제약/NNG', '제외/NNG', '제이앤조이/NNP', '제작/NNG', '제작비/NNG', '제재/NNG', '제정/NNG', '제정’/NNG', '제정안/NNG', '제주/NNP', '제주도/NNP', '제주영상미디어센터/NNP', '제출/NNG', '제품/NNG', '제한/NNG', '제한적/NNG', '제휴사/NNG', '조/NNG', '조/NNP', '조건/NNG', '조경희/NNP', '조금/NNG', '조기/NNG', '조달청/NNG', '조력/NNG', '조례/NNG', '조례안/NNG', '조례제정/NNG', '조리/NNG', '조명/NNG', '조사/NNG', '조사결과/NNG', '조사대상/NNG', '조선/NNP', '조성/NNG', '조세/NNG', '조세현/NNP', '조언/NNG', '조작/NNG', '조정/NNG', '조직/NNG', '조직위/NNG', '조직위원회/NNG', '조직적/NNG', '조치/NNG', '조한진/NNP', '조항/NNG', '존/NNG', '존엄성/NNG', '존재/NNG', '존중/NNG', '졸업/NNG', '졸업생/NNG', '졸업장/NNG', '종/NNG', '종로/NNP', '종로구/NNP', '종료/NNG', '종류/NNG', '종목/NNG', '종목별/NNG', '종사자/NNG', '종일/NNG', '종전/NNG', '종합/NNG', '종합계획/NNG', '종합대책/NNG', '종합병원/NNG', '종합소득/NNG', '종합안내실/NNG', '종합운동장/NNG', '종합적/NNG', '종합지원/NNG', '좌석/NNG', '좌절/NNG', '좌측/NNG', '죄/NNG', '주/NNG', '주간/NNG', '주거/NNG', '주거공간/NNG', '주거복지/NNG', '주거생활/NNG', '주거실태/NNG', '주거실태조사/NNG', '주거약자/NNG', '주거약자지원법/NNG', '주거정책/NNG', '주거지원/NNG', '주거환경/NNG', '주관적/NNG', '주도/NNG', '주말/NNG', '주목/NNG', '주무부처/NNG', '주문/NNG', '주민/NNG', '주민발의/NNG', '주민센터/NNG', '주변/NNG', '주소/NNG', '주식회사/NNG', '주요/NNG', '주위/NNG', '주유소/NNG', '주은미/NNG', '주의/NNG', '주인/NNG', '주인공/NNG', '주장/NNG', '주재/NNG', '주제/NNG', '주제발표/NNG', '주차/NNG', '주차구역/NNG', '주차장/NNG', '주체/NNG', '주체적/NNG', '주최/NNG', '주축/NNG', '주출입구/NNG', '주택/NNG', '주행/NNG', '죽음/NNG', '준결승/NNG', '준결승전/NNG', '준공/NNG', '준비/NNG', '준수/NNG', '줄/NNG', '줄기세포/NNG', '중/NNG', '중간/NNG', '중개기관/NNG', '중계/NNG', '중고/NNG', '중구/NNG', '중국/NNP', '중년/NNG', '중단/NNG', '중도장애인/NNG', '중독/NNG', '중등/NNG', '중랑구/NNP', '중림종합복지센터/NNG', '중립성/NNG', '중복/NNG', '중복사업/NNG', '중복장애/NNG', '중부지방고용노동청/NNG', '중소기업/NNG', '중소기업청/NNG', '중순/NNG', '중심/NNG', '중앙/NNG', '중앙부처/NNG', '중앙선거관리위원회/NNG', '중앙선관위/NNG', '중앙센터/NNG', '중앙일보/NNP', '중앙장애아동지원센터/NNG', '중앙정부/NNG', '중앙정부청사/NNG', '중앙행정기관/NNG', '중앙환원/NNG', '중요성/NNG', '중장기/NNG', '중장기계획/NNG', '중장기적/NNG', '중점/NNG', '중점적/NNG', '중증/NNG', '중증장애/NNG', '중증장애인/NNG', '중증장애인다수고용사업장/NNG', '중증장애인독립생활연대/NNG', '중증장애인생산품/NNG', '중증장애인직업재활지원사업/NNG', '중지/NNG', '중학교/NNG', '중학생/NNG', '중형/NNG', '쥐/NNG', '즉답/NNG', '즉시/NNG', '증가/NNG', '증가율/NNG', '증거/NNG', '증대/NNG', '증상/NNG', '증설/NNG', '증세/NNG', '증액/NNG', '증언/NNG', '증원/NNG', '증인/NNG', '증인신문/NNG', '증인지원/NNG', '증인지원센터/NNG', '증장애인/NNG', '증진/NNG', '증후군/NNG', '지/NNP', '지검/NNG', '지금/NNG', '지급/NNG', '지급기준/NNG', '지급대상/NNG', '지급액/NNG', '지난달/NNG', '지난번/NNG', '지난해/NNG', '지도/NNG', '지도자/NNG', '지름/NNG', '지방/NNG', '지방검찰청/NNG', '지방경찰청/NNG', '지방교육재정교부금/NNG', '지방교육행정기관/NNG', '지방법원/NNG', '지방분권/NNG', '지방세/NNG', '지방세특례/NNG', '지방의회/NNG', '지방자치단체/NNG', '지방자치단체장/NNG', '지방재정/NNG', '지방정부/NNG', '지법/NNG', '지부/NNG', '지부장/NNG', '지사/NNG', '지사장/NNG', '지상/NNG', '지상파/NNG', '지속/NNG', '지속적/NNG', '지식/NNG', '지식경제부/NNG', '지역/NNG', '지역가입자/NNG', '지역공동체/NNG', '지역교육청/NNG', '지역민/NNG', '지역사회/NNG', '지역사회서비스투자사업/NNG', '지역사회자립/NNG', '지역사회재활시설/NNG', '지역사회중심재활사업/NNG', '지역성/NNG', '지역센터/NNG', '지역자원시설세/NNG', '지역장애아동지원센터/NNG', '지역적/NNG', '지역주민/NNG', '지연/NNG', '지원/NNG', '지원계획/NNG', '지원금/NNG', '지원기관/NNG', '지원기준/NNG', '지원대상/NNG', '지원방식/NNG', '지원방안/NNG', '지원사업/NNG', '지원서/NNG', '지원센터/NNG', '지원예산/NNG', '지원자/NNG', '지원정책/NNG', '지원제도/NNG', '지원조례/NNG', '지원체계/NNG', '지원축소/NNG', '지위/NNG', '지인/NNG', '지자체/NNG', '지장/NNG', '지적/NNG', '지적발달/NNG', '지적장애/NNG', '지적장애인/NNG', '지점/NNG', '지정/NNG', '지주/NNG', '지지/NNG', '지청/NNG', '지체/NNG', '지체장애/NNG', '지체장애인/NNG', '지체장애인협회/NNG', '지출/NNG', '지침/NNG', '지팡이/NNG', '지표/NNG', '지하/NNG', '지하도/NNG', '지하수/NNG', '지하철/NNG', '지형/NNG', '직권/NNG', '직권조사/NNG', '직무/NNG', '직업/NNG', '직업교육/NNG', '직업능력개발/NNG', '직업재활/NNG', '직업재활법/NNG', '직업재활서비스/NNG', '직업재활시설/NNG', '직업적/NNG', '직업훈련/NNG', '직원/NNG', '직위/NNG', '직장/NNG', '직장가입자/NNG', '직장생활/NNG', '직장어린이집/NNG', '직장인/NNG', '직전/NNG', '직접적/NNG', '직종/NNG', '직후/NNG', '진/NNP', '진단/NNG', '진로/NNG', '진료/NNG', '진료비/NNG', '진료환자/NNG', '진보/NNG', '진보정의당/NNP', '진보정치/NNG', '진상조사/NNG', '진상조사위원회/NNG', '진선미/NNP', '진술/NNG', '진술조력/NNG', '진술조력인/NNG', '진실/NNG', '진심/NNG', '진입/NNG', '진전/NNG', '진정/NNG', '진정사건/NNG', '진정서/NNG', '진짜/NNG', '진출/NNG', '진행/NNG', '진호/NNP', '질/NNG', '질문/NNG', '질병/NNG', '질병관리본부/NNG', '질적/NNG', '질환/NNG', '짐/NNG', '집/NNG', '집단/NNG', '집단급식소/NNG', '집안/NNG', '집중/NNG', '집중력/NNG', '집중적/NNG', '집행/NNG', '집행부/NNG', '집행위원/NNG', '집행위원장/NNG', '집행유예/NNG', '집회/NNG', '징계/NNG', '징수/NNG', '징역/NNG', '차/NNG', '차관/NNG', '차기/NNG', '차도/NNG', '차등/NNG', '차등화/NNG', '차량/NNG', '차례/NNG', '차별/NNG', '차별금지/NNG', '차별적/NNG', '차별행위/NNG', '차상위/NNG', '차상위계층/NNG', '차오닝닝/NNG', '차오닝닝/NNP', '차원/NNG', '차이/NNG', '차종/NNG', '차질/NNG', '착공/NNG', '찬반/NNG', '참가/NNG', '참가자/NNG', '참고/NNG', '참관단/NNG', '참변/NNG', '참사/NNG', '참석/NNG', '참여/NNG', '참정권/NNG', '창간/NNG', '창구/NNG', '창동계/NNG', '창동계스페셜올/NNP', '창동계스페셜올림픽/NNP', '창립/NNG', '창업/NNG', '창업경진대회/NNG', '창업아이템/NNG', '창업형/NNG', '창원/NNP', '창원시/NNP', '창작/NNG', '창작뮤지컬/NNG', '창출/NNG', '채소/NNG', '채영랑/NNP', '채용/NNG', '채택/NNG', '책/NNG', '책무/NNG', '책임/NNG', '책임감/NNG', '책임자/NNG', '책자/NNG', '책정/NNG', '처리/NNG', '처방전/NNG', '처벌/NNG', '처분/NNG', '처사/NNG', '처우/NNG', '처우개선/NNG', '처음/NNG', '처장/NNG', '척수장애인/NNG', '천막/NNG', '천막농성/NNG', '천안/NNP', '철거/NNG', '철거민/NNG', '첫날/NNG', '청각/NNG', '청각장애/NNG', '청각장애인/NNG', '청구/NNG', '청구소송/NNG', '청년/NNG', '청년드림팀/NNG', '청력/NNG', '청사/NNG', '청사진/NNG', '청소/NNG', '청소년/NNG', '청소년기/NNG', '청와대/NNP', '청원서/NNG', '청원안/NNG', '청중/NNG', '청춘/NNG', '청탁/NNG', '체결/NNG', '체계/NNG', '체계적/NNG', '체급/NNG', '체납/NNG', '체납액/NNG', '체납자/NNG', '체당금/NNG', '체어/NNG', '체육/NNG', '체육관/NNG', '체육시설/NNG', '체육회/NNG', '체전/NNG', '체제/NNG', '체크/NNG', '체험/NNG', '체험홈/NNG', '초/NNG', '초과/NNG', '초기/NNG', '초등/NNG', '초등학교/NNG', '초등학생/NNG', '초반/NNG', '초안/NNG', '초음파/NNG', '초점/NNG', '촉각/NNG', '촉구/NNG', '촉진/NNG', '총력/NNG', '총리/NNG', '총선/NNG', '총선연대/NNG', '총액/NNG', '총장/NNG', '총회/NNG', '촬영/NNG', '최/NNP', '최고/NNG', '최고속도/NNG', '최고점/NNG', '최광근/NNP', '최근/NNG', '최다적발건수/NNG', '최대/NNG', '최대한/NNG', '최동익/NNP', '최선/NNG', '최소/NNG', '최소한/NNG', '최예진/NNP', '최우선/NNG', '최우수등급/NNG', '최우수상/NNG', '최장/NNG', '최저/NNG', '최저기준/NNG', '최저생계비/NNG', '최저임금/NNG', '최저점/NNG', '최저주거기준/NNG', '최적/NNG', '최종/NNG', '최종목적지/NNG', '최종안/NNG', '최종적/NNG', '최초/NNG', '추가/NNG', '추가급여/NNG', '추가비용/NNG', '추가적/NNG', '추가지원/NNG', '추구/NNG', '추락/NNG', '추세/NNG', '추억/NNG', '추정/NNG', '추진/NNG', '추진본부/NNG', '추진위/NNG', '추천/NNG', '추천서/NNG', '추행/NNG', '추후/NNG', '축구/NNG', '축사/NNG', '축소/NNG', '축제/NNG', '축하/NNG', '축하공연/NNG', '춘천/NNP', '출구/NNG', '출마/NNG', '출발/NNG', '출범/NNG', '출범식/NNG', '출산/NNG', '출산비용/NNG', '출생신고/NNG', '출석/NNG', '출신/NNG', '출연/NNG', '출연가수/NNG', '출입/NNG', '출입구/NNG', '출입문/NNG', '출전/NNG', '출정식/NNG', '출처/NNG', '출퇴근/NNG', '충격/NNG', '충남/NNP', '충북/NNP', '충북도/NNP', '충원/NNG', '충전/NNG', '충전시설/NNG', '충정로/NNP', '충족/NNG', '충주시/NNP', '충치/NNG', '충치예방/NNG', '취득/NNG', '취득세/NNG', '취소/NNG', '취소처분/NNG', '취약/NNG', '취약가구/NNG', '취약계층/NNG', '취약지역/NNG', '취업/NNG', '취업기회/NNG', '취업알선/NNG', '취업자/NNG', '취지/NNG', '측근/NNG', '측면/NNG', '층/NNG', '치과/NNG', '치과진료/NNG', '치료/NNG', '치료비/NNG', '치매/NNG', '치아홈메우기/NNG', '친고/NNG', '친구/NNG', '친화/NNG', '침대/NNG', '침해/NNG', '카/NNG', '카드/NNG', '카르텔/NNG', '카메라/NNG', '카운티/NNP', '카테고리/NNG', '카트/NNG', '카페/NNG', '카페베네/NNP', '카페인/NNG', '칼/NNG', '캐나다/NNP', '캐롤라인/NNP', '캘리포니아/NNP', '캠퍼스/NNG', '캠페인/NNG', '캠프/NNG', '캡처/NNG', '커리어/NNG', '커플/NNG', '커피/NNG', '컨디션/NNG', '컨벤시아/NNP', '컨설팅/NNG', '컨퍼런스/NNG', '컬러컨설팅/NNG', '컬럼/NNG', '컴퓨터/NNG', '케이블카/NNG', '코/NNG', '코끼리열차/NNG', '코너/NNG', '코레일/NNG', '코스/NNG', '코스탄티노/NNG', '코올/NNG', '코치/NNG', '콘서트/NNG', '콘텐츠/NNG', '콜택시/NNG', '쾌거/NNG', '크기/NNG', '크레딧/NNG', '큰어금니/NNG', '클래스/NNG', '키워드/NNG', '킨텍스/NNP', '킷츠/NNG', '타이브레이크/NNG', '타인/NNG', '탁구/NNG', '탄력/NNG', '탄력적/NNG', '탈락/NNG', '탈락자/NNG', '탈시설/NNG', '탈원전/NNG', '탈의실/NNG', '탑승/NNG', '탑승설비/NNG', '탓/NNG', '태극기/NNG', '태도/NNG', '태평양/NNP', '태풍/NNG', '택시/NNG', '탤런트/NNG', '터치/NNG', '터치식자동문/NNG', '턱/NNG', '테니스/NNG', '테스트/NNG', '테이블/NNG', '텍스트/NNG', '텔레비전/NNG', '토대/NNG', '토론/NNG', '토론자/NNG', '토론회/NNG', '토론회’/NNG', '토요일/NNG', '통/NNG', '통계/NNG', '통과/NNG', '통로/NNG', '통보/NNG', '통상/NNG', '통역/NNG', '통일/NNG', '통장/NNG', '통증/NNG', '통지/NNG', '통학/NNG', '통학비/NNG', '통학지원/NNG', '통합/NNG', '통합교육/NNG', '통합진보/NNG', '통합진보당/NNP', '통행/NNG', '통행료/NNG', '퇴소/NNG', '퇴원/NNG', '퇴직금/NNG', '투명/NNG', '투입예산/NNG', '투자/NNG', '투쟁/NNG', '투표/NNG', '투표소/NNG', '투표율/NNG', '투표장/NNG', '트럭/NNG', '트레이너/NNG', '트위터/NNG', '특권/NNG', '특단/NNG', '특례/NNG', '특례법/NNG', '특례시험/NNG', '특별/NNG', '특별교통수단/NNG', '특별법/NNG', '특별시/NNG', '특별전형/NNG', '특별채용/NNG', '특보/NNG', '특성/NNG', '특수/NNG', '특수가구/NNG', '특수교사/NNG', '특수교원/NNG', '특수교육/NNG', '특수교육과/NNG', '특수교육교원/NNG', '특수교육기관/NNG', '특수교육대상/NNG', '특수교육대상자/NNG', '특수교육법/NNG', '특수교육지원센터/NNG', '특수교육학과/NNG', '특수성/NNG', '특수학교/NNG', '특수학급/NNG', '특위/NNG', '특정/NNG', '특집/NNG', '특징/NNG', '틀/NNG', '티켓/NNG', '틱장애/NNG', '팀/NNG', '팀원/NNG', '팀장/NNG', '파견/NNG', '파리원칙/NNG', '파악/NNG', '파이팅/NNG', '파주/NNP', '파주시/NNP', '파킨슨병/NNG', '판결/NNG', '판단/NNG', '판매/NNG', '판사/NNG', '판정/NNG', '판정기준/NNG', '팔/NNG', '팝핀현준/NNP', '패널/NNG', '패러다임/NNG', '패럴림픽/NNG', '패스/NNG', '팩스/NNG', '팬/NNG', '퍼레이드/NNG', '퍼스트/NNG', '퍼포먼스/NNG', '펀딩/NNG', '페루/NNP', '페스티벌/NNG', '페이스/NNG', '페이스북/NNG', '펜싱/NNG', '편견/NNG', '편성/NNG', '편의/NNG', '편의시설/NNG', '편의점/NNG', '편의제공/NNG', '편의증진/NNG', '편의증진법/NNG', '편의지원/NNG', '편지/NNG', '편집자주/NNG', '편차/NNG', '평/NNG', '평가/NNG', '평가문항/NNG', '평가지표/NNG', '평가회의/NNG', '평균/NNG', '평균소득/NNG', '평균적/NNG', '평등/NNG', '평상시/NNG', '평생/NNG', '평생교육/NNG', '평소/NNG', '평영/NNG', '평창/NNP', '평창동/NNP', '평창동계스페셜올/NNP', '평창동계스페셜올림픽/NNP', '평택/NNP', '평택시/NNP', '평화/NNG', '폐막식/NNG', '폐지/NNG', '폐지’/NNG', '폐회식/NNG', '포/NNG', '포괄적/NNG', '포도/NNG', '포럼/NNG', '포부/NNG', '포인트/NNG', '포함/NNG', '포항/NNP', '포항시/NNP', '포항시의회/NNP', '폭/NNG', '폭력/NNG', '폭언/NNG', '폭염/NNG', '폭우/NNG', '폭행/NNG', '폴리텍/NNG', '표/NNG', '표시/NNG', '표정/NNG', '표준사업장/NNG', '표지/NNG', '표지판/NNG', '표창/NNG', '표현/NNG', '풀/NNG', '품목/NNG', '품질/NNG', '프랑스/NNP', '프랜드/NNG', '프레지던트호텔/NNP', '프로그램/NNG', '프로젝트/NNG', '프리/NNP', '플래카드/NNG', '플랫폼/NNG', '플로리다/NNP', '피/NNG', '피고인/NNG', '피난/NNG', '피로/NNG', '피부/NNG', '피의자/NNG', '피청구인/NNG', '피켓/NNG', '피폭/NNG', '피해/NNG', '피해여성/NNG', '피해자/NNG', '픽처/NNG', '필기/NNG', '필기시험/NNG', '필름/NNG', '필리핀/NNP', '필수/NNG', '필수예방접종/NNG', '필수적/NNG', '필요/NNG', '필요도/NNG', '필요성/NNG', '필자/NNG', '핑계/NNG', '하/NNG', '하나/NNG', '하늘/NNG', '하락/NNG', '하루/NNG', '하반기/NNG', '하사가장애인상담넷/NNG', '하우스/NNG', '하위/NNG', '하조대/NNP', '하트뮤직스쿨/NNG', '하향/NNG', '학계/NNG', '학과/NNG', '학교/NNG', '학교폭력/NNG', '학교환경/NNG', '학급/NNG', '학년/NNG', '학년도/NNG', '학대/NNG', '학력/NNG', '학부모/NNG', '학부모회/NNG', '학생/NNG', '학생부/NNG', '학술대회/NNG', '학습/NNG', '학장/NNG', '학회장/NNG', '한/NNG', '한/NNP', '한계/NNG', '한곳/NNG', '한국/NNP', '한국농아인협회/NNP', '한국보건복지인력개발원/NNP', '한국성폭력상담소/NNP', '한국어/NNP', '한국장애인개발원/NNP', '한국장애인고용공단/NNP', '한국장애인국제예술단/NNP', '한국장애인단체총연합회/NNP', '한국장애인자립생활센터총연합회/NNP', '한국장총/NNP', '한국정보화진흥원/NNP', '한글/NNG', '한나라당/NNP', '한도/NNG', '한마디/NNG', '한마음/NNG', '한반도/NNP', '한부모가족/NNG', '한센/NNP', '한센인/NNG', '한숨/NNG', '한자연/NNG', '한자연/NNP', '한쪽/NNG', '한편/NNG', '할머니/NNG', '할아버지/NNG', '할인/NNG', '합/NNG', '합격/NNG', '합격자/NNG', '합계/NNG', '합동/NNG', '합리적/NNG', '합병증/NNG', '합의/NNG', '합창단/NNG', '합헌/NNG', '항/NNG', '항고/NNG', '항공기/NNG', '항공사/NNG', '항공전/NNG', '항목/NNG', '항소심/NNG', '항의/NNG', '해/NNG', '해결/NNG', '해단/NNG', '해당/NNG', '해당시설/NNG', '해명/NNG', '해변/NNG', '해석/NNG', '해소/NNG', '해수욕장/NNG', '해양경찰청/NNG', '해외/NNG', '해임/NNG', '핵심/NNG', '핵심공약/NNG', '핵심적/NNG', '핸드레일/NNG', '핸드싸이클/NNG', '핸들/NNG', '행군/NNG', '행군거리/NNG', '행군속도/NNG', '행동/NNG', '행동계획/NNG', '행방/NNG', '행보/NNG', '행복/NNG', '행사/NNG', '행사장/NNG', '행안부/NNG', '행위/NNG', '행위자/NNG', '행정/NNG', '행정기관/NNG', '행정실/NNG', '행정실장/NNG', '행정심판/NNG', '행정안전부/NNG', '행정적/NNG', '행정처분/NNG', '행태/NNG', '향상/NNG', '향유/NNG', '향후/NNG', '허가/NNG', '허락/NNG', '허용/NNG', '허위/NNG', '허정석/NNP', '헌법/NNG', '헌법소원/NNG', '헌법재판관/NNG', '헌법재판소/NNG', '헌신/NNG', '헌재/NNG', '헬기/NNG', '헬기사업자/NNG', '혀/NNG', '혁명/NNG', '혁신적/NNG', '현/NNG', '현금/NNG', '현금급여기준/NNG', '현민/NNG', '현병철/NNP', '현상/NNG', '현실/NNG', '현실”/NNG', '현실성/NNG', '현실적/NNG', '현실화/NNG', '현안/NNG', '현장/NNG', '현장실습/NNG', '현장점검/NNG', '현재/NNG', '현주소/NNG', '현지/NNG', '현지시간/NNG', '현직/NNG', '현행/NNG', '현행법/NNG', '현황/NNG', '혈관/NNG', '혐의/NNG', '협력/NNG', '협박/NNG', '협약/NNG', '협의/NNG', '협의회/NNG', '협조/NNG', '협회/NNG', '형/NNG', '형량/NNG', '형사/NNG', '형성/NNG', '형식/NNG', '형식적/NNG', '형태/NNG', '형편/NNG', '형평성/NNG', '혜택/NNG', '호소/NNG', '호응/NNG', '호주/NNP', '호텔/NNG', '호흡/NNG', '호흡기/NNG', '호흡기장애/NNG', '혹한/NNG', '혼란/NNG', '혼선/NNG', '혼성/NNG', '혼자/NNG', '홈/NNG', '홈페이지/NNG', '홍/NNP', '홍보/NNG', '홍보대사/NNG', '홍보물/NNG', '화/NNG', '화가/NNG', '화두/NNG', '화면/NNG', '화면해설/NNG', '화면해설/NNP', '화요집회/NNG', '화장실/NNG', '화재/NNG', '화제/NNG', '화학적/NNG', '확답/NNG', '확대/NNG', '확률/NNG', '확보/NNG', '확산/NNG', '확인/NNG', '확인결과/NNG', '확인조사/NNG', '확장/NNG', '확정/NNG', '확충/NNG', '환경/NNG', '환급/NNG', '환상/NNG', '환승/NNG', '환영/NNG', '환자/NNG', '활동/NNG', '활동가/NNG', '활동보조/NNG', '활동보조서비스/NNG', '활동보조인/NNG', '활동보조인연대/NNG', '활동지원/NNG', '활동지원급여/NNG', '활동지원서비스/NNG', '활동지원제도/NNG', '활보연대/NNG', '활보제도개선위/NNG', '활성화/NNG', '활약/NNG', '활용/NNG', '황/NNP', '황연대/NNP', '황우여/NNP', '회교육/NNG', '회복/NNG', '회사/NNG', '회원/NNG', '회원국/NNG', '회의/NNG', '회의실/NNG', '회장/NNG', '회전/NNG', '회피/NNG', '획기적/NNG', '획득/NNG', '획일적/NNG', '횟수/NNG', '횡단보도/NNG', '횡령/NNG', '효과/NNG', '효과성/NNG', '효과적/NNG', '효력/NNG', '효율/NNG', '효율적/NNG', '효율적”/NNG', '후/NNG', '후견제/NNG', '후문/NNG', '후반/NNG', '후배/NNG', '후보/NNG', '후보자/NNG', '후보지/NNG', '후속/NNG', '후원/NNG', '후원금/NNG', '후원자/NNG', '후유증/NNG', '후천적/NNG', '후쿠시마/NNP', '후쿠시마현/NNP', '훈련/NNG', '훈련생/NNG', '휠링/NNG', '휠체어/NNG', '휠체어리프트/NNG', '휠체어장애인/NNG', '휴가/NNG', '휴가철/NNG', '휴게소/NNG', '휴게음식점/NNG', '휴대폰/NNG', '휴식/NNG', '휴일/NNG', '휴지/NNG', '휴지걸이/NNG', '흐름/NNG', '흑고래팀/NNG', '흡연/NNG', '흡연실/NNG', '흡연자/NNG', '희귀/NNG', '희망/NNG', '희망자/NNG', '힘/NNG']\n"
     ]
    }
   ],
   "source": [
    "# NNG, NNP만 뽑기\n",
    "\n",
    "# 단어/품사 추출\n",
    "# input: file path\n",
    "# output: list\n",
    "def get_features(features_file):\n",
    "    feature = []\n",
    "    with open(features_file, 'r', encoding='utf=8') as f:\n",
    "        while True:\n",
    "            line = f.readline()\n",
    "            if not line:\n",
    "                break\n",
    "            if line == '\\n':\n",
    "                continue\n",
    "            line = line.strip()\n",
    "            key = line.split('\\t')[1]\n",
    "            feature.append(key)\n",
    "    return feature\n",
    "\n",
    "# 명사 추출\n",
    "# input: 각 문서의 단어가 있는 list\n",
    "# output: 명사류의 list ex) 명사/NNG..\n",
    "def get_noun(feature):\n",
    "    words = []\n",
    "    for f in range(0,len(feature)):\n",
    "        words.append(feature[f].split('+'))\n",
    "    nouns = []\n",
    "    for k in range(0, len(words)):\n",
    "        for w in range(0, len(words[k])):\n",
    "            if '/NNG' in words[k][w]:\n",
    "                nouns.append(words[k][w])\n",
    "            if '/NNP' in words[k][w]:\n",
    "                nouns.append(words[k][w])\n",
    "    return nouns\n",
    "\n",
    "# 각 폴더의 문서 추출\n",
    "# ex) features[0][0] = ['word1', word2 .. ] (0번째 폴더의 0번째 문서의 단어들)\n",
    "features = []\n",
    "for n in range(0, len(input_data)):\n",
    "    tmp = []\n",
    "    for m in range(0, len(input_data[n])):\n",
    "        tmp.append(get_features(input_data[n][m]))\n",
    "    features.append(tmp)\n",
    "\n",
    "# 모든 단어 합치기 - 하나의 리스트로\n",
    "# ex) all_features = [word1, word2 ..]\n",
    "all_features = []\n",
    "for i in range(0, len(features)):\n",
    "    all_features.append(sum(features[i],[]))\n",
    "\n",
    "# 명사들 하나의 리스트로 합치기\n",
    "# ex) nn[0] = ['n1', 'n2', .. ]\n",
    "nn = []\n",
    "for i in range(0, len(all_features)):\n",
    "    nn.append(get_noun(all_features[i]))\n",
    "\n",
    "# 하나의 리스트로 합침\n",
    "# ex) all_noun = [n1, n2, ..]\n",
    "all_noun = sum(nn,[])\n",
    "\n",
    "# 상위 5000 개 추출\n",
    "count_5000 = Counter(all_noun)\n",
    "c = count_5000.most_common(5000)\n",
    "count_dict = dict(c)\n",
    "count_keys = list(count_dict.keys())\n",
    "count_keys.sort()   # 오름차순\n",
    "print(count_keys)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 654,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['장애아동/NNG', '재활치료시설/NNG', '부족/NNG', '장애아동/NNG', '재활치료시설/NNG', '대책/NNG', '마련/NNG', '국회/NNG', '보건복지위원회/NNG', '김희국/NNP', '의원/NNG', '새누리당/NNP', '보건복지부/NNG', '제출/NNG', '자료/NNG', '장애아동/NNG', '반면/NNG', '장애아동/NNG', '재활치료/NNG', '여건/NNG', '김/NNP', '의원/NNG', '국내/NNG', '장애아동/NNG', '전문/NNG', '재활치료/NNG', '곳/NNG', '공공의료기관/NNG', '국립재활원/NNP', '서울시/NNP', '무연고/NNG', '중증장애/NNG', '아동치료/NNG', '보호시설/NNG', '시립아동병원/NNG', '병상/NNG', '곳/NNG', '장애아동/NNG', '재활/NNG', '집중적/NNG', '기관/NNG', '보바스병원/NNP', '곳/NNG', '장애아동/NNG', '진료과/NNG', '재화의학과/NNG', '곳/NNG', '서울시어린이병원/NNP', '부산대/NNP', '부설/NNG', '어린이병원’/NNG', '재활병원/NNG', '곳/NNG', '장애아동/NNG', '전문병원/NNG', '진료/NNG', '실정/NNG', '이외/NNG', '국내/NNG', '어린이전문병원/NNG', '전국/NNG', '비장애아동/NNG', '대상/NNG', '복지부/NNG', '추진/NNG', '권역/NNG', '재활병원/NNG', '건립사업/NNG', '장애아동/NNG', '전문적/NNG', '장애아동/NNG', '재활진료과/NNG', '수준/NNG', '김/NNP', '의원/NNG', '장애아동/NNG', '전문적/NNG', '재활치료/NNG', '문제/NNG', '공공/NNG', '복지서비스/NNG', '필수/NNG', '영역/NNG', '사회적/NNG', '지원체계/NNG', '장애아동/NNG', '전문/NNG', '재활병원/NNG', '추가/NNG', '건립/NNG', '진료과/NNG', '복지부/NNG', '노력/NNG', '의지/NNG', '김/NNP', '의원/NNG', '병원/NNG', '자체/NNG', '결과/NNG', '장애아동/NNG', '조기재활치료/NNG', '장애인/NNG', '인/NNG', '잠재적/NNG', '사회적/NNG', '비용/NNG', '정도/NNG', '장애아동/NNG', '조기재활치료/NNG', '환경/NNG', '병원/NNG', '조사결과/NNG', '장애인/NNG', '의료비/NNG', '지원/NNG', '예산/NNG', '인/NNG', '절감/NNG', '결론/NNG', '장애인/NNG', '지원/NNG', '비용/NNG', '금액/NNG', '설명/NNG', '김/NNP', '의원/NNG', '장애아동/NNG', '장애인/NNG', '재활치료/NNG', '신체/NNG', '정신적/NNG', '경제적/NNG', '회복/NNG', '국가/NNG', '의무”/NNG', '장애/NNG', '아동/NNG', '청소년/NNG', '대부분/NNG', '장애특성/NNG', '발달/NNG', '재활치료/NNG', '의료/NNG', '교육/NNG', '사회/NNG', '심리적/NNG', '영역/NNG', '내용/NNG', '장애아동/NNG', '청소년/NNG', '가족/NNG', '원조/NNG', '교육/NNG']\n"
     ]
    }
   ],
   "source": [
    "# 각 폴더의 명사 ..\n",
    "# ex) noun[0][0] = [김/NNP', '의원/NNG' .. ]  :: 0번째 폴더의 0번째 파일의 단어..\n",
    "noun = []\n",
    "for i in range(0, len(features)):\n",
    "    temp_noun = []\n",
    "    for j in range(0, len(features[i])):\n",
    "        temp_noun.append(get_noun(features[i][j]))\n",
    "    noun.append(temp_noun)\n",
    "\n",
    "print(noun[0][0])"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### 3. calculate TF\n",
    "    - all_TF[][][] : folder, doc, 5000..\n",
    "    - ex) all_TF[0][0] : child의 child_1.txt 의 5000단어 tf\n",
    "    - ex) all_TF[0][0][0] == 9 -> child의 child_1.txt의 상위 5000 중 첫번째 단어의 tf"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 655,
   "outputs": [],
   "source": [
    "# tf\n",
    "# ex) tf[0][0] = child 폴더의 첫번째 파일의 tf 모음 (5000개)\n",
    "# ex) tf[0][0][0] = child 폴더의 첫번째 파일의 5000개중 첫번재 단어의 tf\n",
    "def calculate_TF(nouns_data, keys):\n",
    "    tf = []\n",
    "    for nd in range(0, len(nouns_data)):\n",
    "        tmp_tf = []\n",
    "        for md in range(0, len(nouns_data[nd])):\n",
    "            tmp_tf2 = []\n",
    "            for k in range(0, len(keys)):\n",
    "                t = [s for s in nouns_data[nd][md] if keys[k] in s]\n",
    "                tmp_tf2.append(len(t))\n",
    "            tmp_tf.append(tmp_tf2)\n",
    "        tf.append(tmp_tf)\n",
    "\n",
    "    return tf\n",
    "\n",
    "all_tf = calculate_TF(noun, count_keys)\n"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": 656,
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['전국/NNG', '시각장애인/NNG', '안마사/NNG', '거리/NNG', '시각장애인/NNG', '안마사/NNG', '의료법/NNG', '시각장애인/NNG', '안마사/NNG', '자격/NNG', '합헌결정/NNG', '거리/NNG', '대한/NNP', '안마사협회/NNG', '회원/NNG', '오전/NNG', '여의도/NNP', '국회/NNG', '국민은행/NNP', '앞/NNG', '안마사자격제도/NNG', '합헌/NNG', '기원/NNG', '결의대회’/NNG', '이번/NNG', '결의대회/NNG', '헌법재판관/NNG', '후보자/NNG', '국회/NNG', '인사청문회/NNG', '시각장애인/NNG', '안마사자격/NNG', '관심/NNG', '합헌/NNG', '촉구/NNG', '목소리/NNG', '국회/NNG', '인사청문회/NNG', '헌법재판관/NNG', '후보자/NNG', '대법원장/NNG', '김창종/NNP', '이진성/NNP', '국회/NNG', '몫/NNG', '안창호/NNP', '새누리당/NNP', '지명/NNP', '김이수/NNP', '민주당/NNP', '지명/NNG', '강일원/NNP', '여야/NNP', '공동/NNG', '지명/NNP', '후보/NNG', '국회/NNG', '인사청문회/NNG', '헌재/NNG', '계류/NNG', '시각장애인/NNG', '안마사/NNG', '위헌법률심판/NNG', '헌법소원/NNG', '심리/NNG', '헌재/NNG', '지난해/NNG', '서울중앙/NNP', '지방법원/NNG', '시각장애인/NNG', '안마사제도/NNG', '의료법/NNG', '헌법/NNG', '소지/NNG', '위헌법률심판/NNG', '계류/NNG', '당시/NNG', '서울/NNP', '지법/NNG', '위헌심판/NNG', '제청/NNG', '이유/NNG', '생계/NNG', '일반국민/NNG', '차별’/NNG', '안마사/NNG', '선택권부재/NNG', '소비자/NNG', '행복추구/NNG', '마사지학과/NNG', '마사지사/NNG', '직업선택/NNG', '제한’/NNG', '스포츠마사지업계/NNG', '안마사자격제도/NNG', '헌법손원/NNG', '계류/NNG', '대한/NNP', '안마사협회/NNG', '이병돈/NNP', '회장/NNG', '시각장애인/NNG', '직업/NNG', '안/NNG', '꿈/NNG', '희망/NNG', '생명”/NNG', '합헌/NNG', '판결/NNG', '시각장애인/NNG', '생존권/NNG', '강학자/NNP', '수석부회장/NNG', '안마사제도/NNG', '시각장애인/NNG', '정부/NNG', '국민/NNG', '복지제도/NNG', '고용제도/NNG', '무자격/NNG', '시각장애인/NNG', '안마행위자/NNG', '업권/NNG', '상태”/NNG', '헌재/NNG', '합헌/NNG', '결정/NNG', '자신/NNG', '행위/NNG', '합법적/NNG', '영업행위/NNG', '재심판/NNG', '청구/NNG', '정부/NNG', '시각장애인/NNG', '곽경환/NNP', '부회장/NNG', '직장/NNG', '지금/NNG', '안마사/NNG', '딸/NNG', '가족/NNG', '생계/NNG', '시각장애인/NNG', '안마/NNG', '직업”/NNG', '안마사협회/NNG', '안마사자격/NNG', '합헌/NNG', '판결/NNG', '무자격/NNG', '안마사/NNG', '처벌/NNG', '이하/NNG', '침/NNG', '시술권리/NNG', '보장/NNG', '무차별적/NNG', '안마시술/NNG', '단속/NNG', '중단/NNG', '요양시설/NNG', '보건소/NNG', '안마사/NNG', '확대/NNG', '대한/NNP', '안마사협회/NNG', '회원/NNG', '오전/NNG', '여의도/NNP', '국회/NNG', '국민은행/NNP', '앞/NNG', '안마사자격제도/NNG', '합헌/NNG', '기원/NNG', '결의대회/NNG', '참석/NNG', '분위기/NNG', '대한/NNP', '안마사협회/NNG', '회원/NNG', '여의도국회/NNP', '국민은행/NNP', '앞/NNG', '안마사자격제도/NNG', '합헌/NNG', '기원/NNG', '결의대회’/NNG', '정부/NNG', '각성/NNG', '대한/NNP', '안마사협회/NNG', '회원/NNG', '위헌제청/NNG', '왠말/NNG', '법복/NNG', '문구/NNG', '피켓/NNG', '대한/NNP', '안마사협회/NNG', '회원/NNG', '안마업권/NNG', '인복지/NNG', '내용/NNG', '피켓/NNG', '대한/NNP', '안마사협회/NNG', '안마사자격제도/NNG', '합헌/NNG', '기원/NNG', '결의대회/NNG', '현수막/NNG', '모습/NNG']\n",
      "Corpus/Input_Data/economy/9_(POS)economy_166.txt\n"
     ]
    }
   ],
   "source": [
    "# noun, input_data 의 같은 index가 같은 파일을 가리키는지 확인\n",
    "print(noun[2][165])\n",
    "print(input_data[2][165])"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### 4. calculate IDF"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# idf\n",
    "# ex) idf[0][0] = child 폴더의 첫번째 파일의 idf 모음 (5000개)\n",
    "# ex) idf[0][0][0] = child 폴더의 첫번째 파일의 5000개중 첫번재 단어의 idf\n",
    "\n",
    "# get number of docs (문서 개수 가져오기)\n",
    "def get_number_of_docs(doc):\n",
    "    N = 0\n",
    "    for d in range(0, len(doc)):\n",
    "        N += len(doc[d])\n",
    "    return N\n",
    "\n",
    "# df 계산\n",
    "def calculate_DF(nouns_data, keys):\n",
    "    df = []\n",
    "    for k in range(0, len(keys)):\n",
    "        num = 0\n",
    "        for nd in range(0, len(nouns_data)):\n",
    "            tmp_df = [d for d in nouns_data[nd] if keys[k] in d]\n",
    "            num += len(tmp_df)\n",
    "        df.append(num)\n",
    "    return df\n",
    "\n",
    "# idf 계산\n",
    "def calculate_IDF(df, N):\n",
    "    idf = []\n",
    "    for d in range(0, len(df)):\n",
    "        d_tmp = (df[d] + 1)\n",
    "        idf_tmp = N / d_tmp\n",
    "        idf.append(np.log10(idf_tmp))\n",
    "    return idf\n",
    "\n",
    "doc_n = get_number_of_docs(input_data)\n",
    "all_df = calculate_DF(noun, count_keys)\n",
    "all_idf = calculate_IDF(all_df, doc_n)\n"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### 5. Calculate TF-IDF"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# TF-IDF\n",
    "# ex) tfidf[0][0] = child 폴더의 첫번째 파일의 tfidf 모음 (5000개)\n",
    "# ex) tfidf[0][0][0] = child 폴더의 첫번째 파일의 5000개중 첫번재 단어의 tfidf\n",
    "\n",
    "def calculate_TFIDF(tf, idf):\n",
    "    TF = np.array(tf)\n",
    "    IDF = np.array(idf)\n",
    "\n",
    "    TFIDF = []\n",
    "    for t in range(0, len(tf)):\n",
    "        tmp_tfidf = []\n",
    "        for f in range(0, len(tf[t])):\n",
    "            tfidf = TF[t][f] * IDF\n",
    "            tmp_tfidf.append(list(tfidf))\n",
    "        TFIDF.append(tmp_tfidf)\n",
    "\n",
    "    return TFIDF\n",
    "\n",
    "all_tfidf = calculate_TFIDF(all_tf, all_idf)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "#### Make my input data path ( 202035535_Leejiyun/child/.. .txt )"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# path setting\n",
    "# ex) 202035535_leejiyun/Input_Data/education/13_(POS)education_3.txt\n",
    "def create_data_path(uid, data):\n",
    "    p = data.split('/')\n",
    "    tmp_path = ''\n",
    "    for len_p in range(1, len(p)):\n",
    "        tmp_path += '/' + str(p[len_p])\n",
    "\n",
    "    path = uid + tmp_path\n",
    "    return path\n",
    "\n",
    "my_input_data_path = []\n",
    "for i in range(0, len(input_data)):\n",
    "    tmp_input_data = []\n",
    "    for j in range(0, len(input_data[i])):\n",
    "        tmp_input_data_2 = create_data_path(student_id, input_data[i][j])\n",
    "        tmp_input_data.append(tmp_input_data_2)\n",
    "    my_input_data_path.append(tmp_input_data)\n",
    "\n",
    "print(my_input_data_path[3][2])"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### 6. Write files"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Write files : TF-IDF\n",
    "def write_files(path, tfidf):\n",
    "    for p in range(0, len(path)):\n",
    "        for q in range(0, len(path[p])):\n",
    "            f = open(path[p][q], 'w')\n",
    "            l = list(tfidf[p][q])\n",
    "            join_list = '\\t'.join(map(str, l))\n",
    "            result = join_list + '\\t' + str(p)\n",
    "            f.write(result)\n",
    "    f.close()\n",
    "\n",
    "write_files(my_input_data_path, all_tfidf)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "# 1.2\n",
    " - from Test_Data\n",
    "    1. Extract nouns from test data\n",
    "    2. Extraction of 5000 least frequently\n",
    "    3. calculate TF\n",
    "    4. calculate IDF\n",
    "    5. calculate TF-IDF\n",
    "    6. Write files"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# set test data\n",
    "test_root = 'Corpus/Test_Data'\n",
    "test_dir_path = []\n",
    "\n",
    "# get search directory path & file path\n",
    "search_dir(test_root, test_dir_path)\n",
    "test_data = []\n",
    "for i in range(0,len(test_dir_path)):\n",
    "    test_data_path = []\n",
    "    search_files(test_dir_path[i], test_data_path)\n",
    "    test_data.append(test_data_path)\n",
    "\n",
    "print(test_dir_path)\n",
    "print(test_data)\n",
    "\n",
    "# create folder\n",
    "for i in range(0, len(test_dir_path)):\n",
    "    test_temp_path = test_dir_path[i].split('/')\n",
    "    test_path_key = test_temp_path[1]+ '/' + test_temp_path[2]\n",
    "    create_folder(student_id + '/' + str(test_path_key))\n",
    "\n",
    "\n",
    "# get features\n",
    "test_features = []\n",
    "for n in range(0, len(test_data)):\n",
    "    test_tmp = []\n",
    "    for m in range(0, len(test_data[n])):\n",
    "        test_tmp.append(get_features(test_data[n][m]))\n",
    "    test_features.append(test_tmp)\n",
    "\n",
    "# 모든 단어 합치기 - 하나의 리스트로\n",
    "# ex) all_features = [word1, word2 ..]\n",
    "test_all_features = []\n",
    "for i in range(0, len(test_features)):\n",
    "    test_all_features.append(sum(test_features[i],[]))\n",
    "\n",
    "# 명사들 하나의 리스트로 합치기\n",
    "# ex) nn[0] = ['n1', 'n2', .. ]\n",
    "test_nn = []\n",
    "for i in range(0, len(test_all_features)):\n",
    "    test_nn.append(get_noun(test_all_features[i]))\n",
    "\n",
    "# get nouns\n",
    "test_noun = []\n",
    "for i in range(0, len(test_features)):\n",
    "    test_temp_noun = []\n",
    "    for j in range(0, len(test_features[i])):\n",
    "        test_temp_noun.append(get_noun(test_features[i][j]))\n",
    "    test_noun.append(test_temp_noun)\n",
    "\n",
    "# calculate TF\n",
    "test_all_tf = calculate_TF(test_noun, count_keys)\n",
    "\n",
    "# get number of documents\n",
    "test_doc_n = get_number_of_docs(test_data)\n",
    "# calculate DF\n",
    "test_all_df = calculate_DF(test_noun, count_keys)\n",
    "# calculate IDF\n",
    "test_all_idf = calculate_IDF(test_all_df, test_doc_n)\n",
    "\n",
    "# calculate TF-IDF\n",
    "test_all_tfidf = calculate_TFIDF(test_all_tf, test_all_idf)\n",
    "\n",
    "# test data path setting\n",
    "my_test_data_path = []\n",
    "for i in range(0, len(test_data)):\n",
    "    tmp_test_data = []\n",
    "    for j in range(0, len(test_data[i])):\n",
    "        tmp_test_data_2 = create_data_path(student_id, test_data[i][j])\n",
    "        tmp_test_data.append(tmp_test_data_2)\n",
    "    my_test_data_path.append(tmp_test_data)\n",
    "\n",
    "# Write files :: TF-IDF\n",
    "write_files(my_test_data_path, test_all_tfidf)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "# 1.3 & 1.4\n",
    "- write train features files & test features files"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Write files :: TFIDF .. + category number\n",
    "def write_features_files(path, tfidf, data_type):\n",
    "    name = f'all_{data_type}_features'\n",
    "    f = open(name, 'w')\n",
    "    for p in range(0, len(path)):\n",
    "        for q in range(0, len(path[p])):\n",
    "            l = list(tfidf[p][q])\n",
    "            join_list = '\\t'.join(map(str, l))\n",
    "            result = join_list + '\\t' + str(p) + '\\n'\n",
    "            f.write(result)\n",
    "    f.close()\n",
    "\n",
    "write_features_files(input_data,all_tfidf, 'train')\n",
    "write_features_files(test_data,test_all_tfidf, 'test')\n"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "# 2.1\n",
    "----\n",
    "### Make train X,y & test X, y"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# get X, y ( for train, test )\n",
    "def get_XY(data):\n",
    "    X = []\n",
    "    y = []\n",
    "    for i in range(0, len(data)):\n",
    "        for j in range(0, len(data[i])):\n",
    "            X.append(data[i][j])\n",
    "            y.append(i)\n",
    "\n",
    "    return X, y\n",
    "\n",
    "# Create train X, train y, test X, test y\n",
    "train_X, train_y = get_XY(all_tfidf)\n",
    "test_X, test_y = get_XY(test_all_tfidf)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### Train SVM model"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Train SVM Model\n",
    "svm_model = svm.SVC(kernel='linear')\n",
    "svm_model.fit(train_X, train_y)\n",
    "\n",
    "# predict\n",
    "pred_y = svm_model.predict(test_X)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "# 2.2\n",
    "---\n",
    "### Confusion Matrix"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Visualization Confusion Matrix\n",
    "def visualization_confusion_matrix(model, test_x, true_y, label):\n",
    "    plot = plot_confusion_matrix(model, test_x, true_y, display_labels=label, cmap=plt.cm.Blues)\n",
    "    plot.ax_.set_title('Confusion Matrix')\n",
    "\n",
    "# create labels :: ex) 0,1, ... ,7,8\n",
    "labels = []\n",
    "for i in range(0,len(all_tfidf)):\n",
    "    labels.append(i)\n",
    "\n",
    "\n",
    "visualization_confusion_matrix(svm_model, test_X, test_y, labels)\n"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "markdown",
   "source": [
    "### Get Precision, Recall, F1 Score"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%% md\n"
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [
    "# Get precision, recall, f1 scores\n",
    "def get_scores(true, pred):\n",
    "    avr = 'micro'\n",
    "    score = {'precision_score': precision_score(true, pred, average=avr),\n",
    "             'recall_score': recall_score(true, pred, average=avr),\n",
    "             'f1_score': f1_score(true, pred, average=avr)}\n",
    "\n",
    "    return score\n",
    "\n",
    "# Visualization Scores\n",
    "def visualization_scores(score):\n",
    "    x = []\n",
    "    h = []\n",
    "    for key, value in score.items():\n",
    "        print(key + \": \" + str(value))\n",
    "        x.append(key)\n",
    "        h.append(value)\n",
    "    x.reverse()\n",
    "    h.reverse()\n",
    "\n",
    "    plt.barh(x, h)\n",
    "    plt.title('Scores')\n",
    "    plt.show()\n",
    "\n",
    "# print Result (precision score, recall score, f1 score)\n",
    "scores = get_scores(test_y, pred_y)\n",
    "print('---Result---')\n",
    "visualization_scores(scores)"
   ],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n",
     "is_executing": true
    }
   }
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "outputs": [],
   "source": [],
   "metadata": {
    "collapsed": false,
    "pycharm": {
     "name": "#%%\n"
    }
   }
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}