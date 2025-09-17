# mlops/feature_engineering/preprocessor_audio_databricks
#from databricks.sdk.runtime import *
import os
import warnings
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
#import pyspark as spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, StringType, ArrayType, FloatType
import sys
#from databricks.sdk.runtime import *
#notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().#notebookPath().get())
#os.chdir(notebook_path)
#os.chdir('..')
#sys.path.append("../..")
from mlops.utils.logging_utils import setup_logger
from mlops.utils.config import DatabricksDataConfig

class FrogEmbedDatabricks:
    """
    ---------------------------------------------------------------------------
    FrogEmbedDatabricks
    ---------------------------------------------------------------------------
    functions associated with sotring and retrieving data from the databricks table
    fns include:
            - storing
            - checking for already existing
            - retrieving datasets for the model training or testing
            - other functionality that will be helpful

    Example:
        config = DatabricksConfig()
        embed_extractor = FrogEmbedDatabricks(config)
        y = embed_extractor.store_on_databricks(df_results)
    """
    
    def __init__(self, config: DatabricksDataConfig):
        """
        Config looks like this
        table: str = "aus_museum_dbx_dev.frogid_ml.dev_embed_table"
        overlap_duration: float = 0.0
        embeddings: str = 'BirdNet'
        embedding_strategy: str = 'averaging'/'stack'/'strong'
        window_duration: float = 3.0
        """

        # General config for the databricks as shown above
        self.config = config
        # Setup general logging
        self.logger = setup_logger(name = self.__class__.__name__)  




    def store_on_databricks(self, df_results: pd.DataFrame) -> bool:
        """
        Function to store the final dataset of embeddings that have just been created to the database
        INPUT:
                self - the class object which has the parameters we also want to include in the storing of data
                df_results - the dataframe of embeddings that have just been created
                            id | chunk_index | features | class_label | species_name
        OUTPUT:
                storing of the embeddings into the table aus_museum_dbx_dev.frogid_ml.dev_embed_table
                audio_id (str) | chunk_index (str) | embedding (ARRAY<float>) | /
                    embedding_strategy (str) | species_name (str) | embedding_from (str) | window_sec (float) | /
                    overlap_window_sec (float)

        """
        #Check that the data being stored isn't already there (caretaker function)
        print(df_results)
        intersection_df, missing_df = self.data_embeddings_search(df_results)
        if missing_df.empty:
            self.logger.info("All data exists in the table. No data added.")
            return False
        if len(intersection_df) > 0 :
            self.logger.info("Some data already exists in the table. Adding the missing data to the database.")
            df_results = missing_df
        print("Storing the embeddings into the table")
        try:
            spark_df = SparkSession.builder.getOrCreate().createDataFrame(df_results)
            spark_df = spark_df.withColumnRenamed("id","audio_id")
            spark_df = spark_df.withColumn("audio_id", spark_df["audio_id"].cast(StringType()))
            spark_df = spark_df.withColumn("chunk_index", spark_df["chunk_index"].cast(IntegerType()))
            spark_df = spark_df.withColumn("species_name", spark_df["species_name"].cast(StringType()))
            spark_df = spark_df.drop("class_label")
            spark_df = spark_df.withColumnRenamed("features","embedding")
            spark_df = spark_df.withColumn("embedding_strategy", lit(self.config.embedding_strategy))
            spark_df = spark_df.withColumn("embedding_from", lit(self.config.embeddings))
            spark_df = spark_df.withColumn("window_sec", lit(self.config.window_duration))
            spark_df = spark_df.withColumn("window_sec", spark_df["window_sec"].cast(FloatType()))
            spark_df = spark_df.withColumn("overlap_window_sec", lit(self.config.overlap_duration)) 
            spark_df = spark_df.withColumn("overlap_window_sec", spark_df["overlap_window_sec"].cast(FloatType()))
            spark_df.write.mode("append").partitionBy("audio_id", "species_name").saveAsTable(self.config.table)
            return True
        except Exception as e:
            self.logger.error(f"❌ embeddings data unable to be stored in {self.config.table}: {e}")
            raise RuntimeError(f"Failed to load audio: {e}")
            
            return False

    def data_embeddings_search(self, df_modelling: pd.DataFrame):
        # get the ids from the table
        #get the ids - create a temporary table in spark
        dt = df_modelling["id"].dtype
        list_of_ids = df_modelling["id"].tolist()
        string_list = [str(num) for num in list_of_ids] #the table stores the ids as strings - no gaurantee the ids will always be numbers
        tup_list = tuple(string_list)
        
        SQL = f"""
        SELECT DISTINCT audio_id
        FROM {self.config.table}
        WHERE embedding_strategy = '{str(self.config.embedding_strategy)}'
        AND embedding_from = '{str(self.config.embeddings)}'
        AND audio_id IN {tup_list}
        AND window_sec = {self.config.window_duration}
        AND overlap_window_sec = {self.config.overlap_duration}
        """
        spark = SparkSession.builder.appName("SQL_Execution").getOrCreate()
        df = spark.sql(SQL)
        df = df.toPandas()
        df = tuple(df["audio_id"].tolist())

        #intersection
        intersection_set = set(tup_list).intersection(df)
        intersection_df = pd.DataFrame(tuple(intersection_set), columns=["id"])
        intersection_df["id"] = intersection_df["id"].astype(dt) #need to match the datatypes of joining columns - pandas errors otherwise
        
        #minus
        missing_set = set(tup_list).difference(df)
        missing_df = pd.DataFrame(tuple(missing_set), columns=["id"])
        missing_df["id"] = missing_df["id"].astype(dt) #need to match the datatypes of joining columns - pandas errors otherwise

        #Now we can return the df_modelling table based on the different sets of ids
        intersection_df = pd.merge(df_modelling, intersection_df, on="id", how="inner")
        missing_df = pd.merge(df_modelling, missing_df, on="id", how="inner")
    
        return intersection_df, missing_df


    def data_embeddings_retrieve(self, df_modelling: pd.DataFrame) -> pd.DataFrame:
        # get the ids from the table
        #get the ids - create a temporary table in spark
        dt = df_modelling["id"].dtype
        list_of_ids = df_modelling["id"].tolist()
        string_list = [str(num) for num in list_of_ids] #the table stores the ids as strings - no gaurantee the ids will always be numbers
        tup_list = tuple(string_list)
        
        SQL = f"""
        SELECT audio_id AS id, chunk_index, embedding as features, species_name
        FROM {self.config.table}
        WHERE embedding_strategy = '{str(self.config.embedding_strategy)}'
        AND embedding_from = '{str(self.config.embeddings)}'
        AND audio_id IN {tup_list}
        AND window_sec = {self.config.window_duration}
        AND overlap_window_sec = {self.config.overlap_duration}
        """
        spark = SparkSession.builder.appName("SQL_Execution").getOrCreate()
        df = spark.sql(SQL)
        df = df.toPandas()
        
        #now we need to join the embedding and chunk_index columns to the df_modelling table
        df["id"] = df["id"].astype(dt)
        df["chunk_index"] = df["chunk_index"].astype('int64')

        return_df = pd.merge(df_modelling, df, on="id", how="inner")
        return_df = return_df[["id","chunk_index","features","class_label_single","species_name"]]
        return_df = return_df.rename(columns={"class_label_single":"class_label"})

        return return_df
    