import pandas as pd
import numpy as np
import yaml
import hashlib
import base64  
import unicodedata
import json
import os
import ast
import openpyxl
def read_excel(file_path):
    return pd.read_excel(file_path)
def rename_Data_Frame(df,rename_col_nm):
    df.rename(columns = rename_col_nm, inplace = True)
    return df
def Add_header_Data_Frame(df,header):
    df.columns = header
    return df
def column_drop(df,col_drop):
    df.drop(col_drop, axis = 1, inplace = True)
    return df
def myhash(x,col):
            key_hash=""           
            if(len(col)==0):
                strhash=str(x.values).replace('nan','No Data found')
                strhash1=strhash.upper()
                hexa = hashlib.sha224(strhash1.encode('utf-8')).hexdigest()               
            else:
                for value in col:
                    col=(str(x[value])).strip().replace('nan','No Data found')
                    key_hash = key_hash + col
                hexa = hashlib.sha224(key_hash.encode('utf-8')).hexdigest()
            return hexa
def Extracting_Duplicate_Value(df,key_col_list,file_path):
   duplicates_df=df[df.duplicated(key_col_list, keep=False)]
   if not duplicates_df.empty:
       if(key_col_list=='Src_Key'):
            df=df.drop_duplicates(subset=['Src_Key'])
            file_path=file_path+"Src_Duplicate.csv"
            duplicates_df.to_csv(file_path,index=False)
            return df
       else:
           df=df.drop_duplicates(subset=['Tgt_Key'])
           file_path=file_path+"Tgt_Duplicate.csv"
           duplicates_df.to_csv(file_path,index=False)
           return df
   else:
         return df 
def fn_err(x,sdf,fd,key):
    err_col = ""
    sn=len(sdf.columns)
    tn=len(fd.columns)
    off=""
    value=""
    final=""
    for item in key:
            item=item+'_x'
            final=final+item+','
            value=value+x[item]+','
    for i in range(1,sn):
        if(str(x[fd.columns[i]]).replace ('nan','No Data found') != str(x[fd.columns[sn+i]]).replace ('nan','No Data found')):
            err_col =err_col+str(fd.columns[i])+' = '+str(x[fd.columns[i]])+' And '+str(fd.columns[sn+i])+' = '+str(x[fd.columns[sn+i]])+','
        else:
          continue
    if(len(err_col) == 0):
        err_col = "PASSED"
    else:
        final=str(key)
        final=final.replace("['",'').replace("']",'')
        err_col = 'Failed: Descripancy (Target Vs Source) for columns - '+  err_col +' And '+ str(final)+'=' +str(value)
    return err_col 
def error_report_generation_with_key_col(join_df,file_loactiton1,file_loactiton2,key_col_list,src_df):
    Mfltr_df = join_df[join_df['_merge']=='both']
    if Mfltr_df.empty:
        pass
    else:
        Mfltr_df['Error_Message'] = Mfltr_df.apply(lambda x:fn_err(x,src_df,Mfltr_df,key_col_list),axis = 1)
        Mfltr_df.to_csv(file_loactiton1,sep=',', encoding='utf-8', header=True, index=False)
    NMfltr_df = join_df[join_df['_merge']!='both']
    if NMfltr_df.empty:
        pass
    else:
       NMfltr_df.to_csv(file_loactiton2,sep=',', encoding='utf-8', header=True, index=False)
    
def main():
    try:
     #Validate configuration paramter correctly set-up in config yaml file
        with open('FileRecon_Config.yml', 'r') as fl:
            param = yaml.safe_load(fl)
        if(len(param['SOURCE_FILE_LOC']) == 0):
           raise Exception("Input Validation Failed: Source File Location Can't Be Blank")
        elif (len(param['SOURCE_FILE_LOC']) != 0 and len(param['SOURCE_FILE_NAME']) == 0):
           raise Exception("Input Validation Failed: Source File Name Can't be Blank")
        elif(len(param['TGT_FILE_LOC']) == 0):
           raise Exception("Input Validation Failed: Target File Location Can't be Blank")
        elif (len(param['TGT_FILE_LOC']) != 0 and len(param['TGT_FILE_NAME']) == 0):
            raise Exception("Input Validation Failed: Target File Name Can't be Blank")
        try:
            usecols = ast.literal_eval(param['USECOLS'])
            rename_col_nm=ast.literal_eval(param['RENAME_COL_NAMES'])
            colspecs = ast.literal_eval(param['COLSPECS'])
            collist = ast.literal_eval(param['NEW_COL_NAMES'])
            dtype = ast.literal_eval(str(param['DTYPE']))
            key_col_list = param['RECON_KEY_COLUMN'] 
            if(param['FILE_IDENTIFER'] == 0):
                #src_df = pd.read_csv(param['SOURCE_FILE_LOC']+param['SOURCE_FILE_NAME'],sep=param['SOURCE_FILE_DELIM'],header=param['TOP_HEADER'],usecols=usecols,skiprows = param['ROWSKIP'],skipfooter= param['FOOT_SKIP'],dtype = param['DTYPE'])
                #tgt_df = pd.read_csv(param['TGT_FILE_LOC']+param['TGT_FILE_NAME'],sep=param['TGT_FILE_DELIM'],header=param['TOP_HEADER'],usecols=usecols,skiprows = param['ROWSKIP'],skipfooter= param['FOOT_SKIP'],dtype = param['DTYPE'])
                src_df=read_excel('C:/Users/U1002900/Desktop/python Code/DataHub_Validation_Opella/Src_File_Try.xlsx') ## please replace the src_file_path and add your source_file_path
                tgt_df=read_excel('C:/Users/U1002900/Desktop/python Code/DataHub_Validation_Opella/Tgt_File_Url.xlsx')   
            else:
                src_df = pd.read_fwf(param['SOURCE_FILE_LOC']+param['SOURCE_FILE_NAME'],header=param['TOP_HEADER'],colspecs=colspecs,usecols=usecols,skiprows = param['ROWSKIP'],skipfooter= param['FOOT_SKIP'],dtypes = param['DTYPE'],nrows = None)
                tgt_df = pd.read_fwf(param['TGT_FILE_LOC']+param['TGT_FILE_NAME'],colspecs=colspecs,names=collist,usecols=usecols,skiprows = param['ROWSKIP'],skipfooter= param['FOOT_SKIP'],dtypes = param['DTYPE'], nrows = None)
            if(param['TOP_HEADER'] != 0):
                header=ast.literal_eval(param['TOP_HEADER'])
                src_df=Add_header_Data_Frame(src_df,header)
                tgt_df=Add_header_Data_Frame(tgt_df,header)
            else:
                pass
            if(param['RENAME_FLAG'] == 'Y'):
                src_df=rename_Data_Frame(src_df,rename_col_nm)
                tgt_df=rename_Data_Frame(tgt_df,rename_col_nm)
            else:
                pass
            if(len(param['DF_DROP']) != 0):
                src_df=column_drop(src_df,param['DF_DROP'])
                tgt_df=column_drop(tgt_df,param['DF_DROP'])
            else:
                pass
            src_df['Src_Key'] =  src_df.apply(lambda x:myhash(x,key_col_list),axis = 1)
            tgt_df['Tgt_Key'] =  tgt_df.apply(lambda x:myhash(x,key_col_list),axis = 1)
            #src_df=Extracting_Duplicate_Value(src_df,"Src_Key",param['SRC_FILE_LOC'])
            #tgt_df=Extracting_Duplicate_Value(tgt_df,"Tgt_Key",param['TGT_FILE_LOC'])
            join_df = pd.merge(src_df, tgt_df,  how='outer', left_on=['Src_Key'], right_on = ['Tgt_Key'],indicator=True)
            print(join_df)
            if(len(key_col_list)!=0):
                error_report_generation_with_key_col(join_df,param['ERROR_REPORT1'],param['ERROR_REPORT2'],key_col_list,src_df)         
            print("Done")
        except Exception as e:
            print(e)
    except Exception as e:
       print(e)
     
if __name__ == '__main__':
    main()
    