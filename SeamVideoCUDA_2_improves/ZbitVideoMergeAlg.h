//--------------------------------------------------------------------
// �Ϻ��������ݿƼ����޹�˾
// 2017��4��7��
// ���ƣ�����
// ��ˣ�
// �޸ļ�¼��2017��4��7��
// 1)
//--------------------------------------------------------------------
// BUG ��¼:
//
//--------------------------------------------------------------------

//
// TODO: 1) ��������������Լ��ṹ����޸�  ���
//      
//

#ifndef __VIDEO_MERGE_ALG_H__
#define __VIDEO_MERGE_ALG_H__

#ifdef WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif



#include "StitchAndSeam.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IN
#define OUT

#define Video_Merge_Handler void*

typedef struct zbit_vm_cfg
{
	int m_iInputVideoWidth;
	int m_iInputVideoHeight;
	int m_iVideoType;
	int m_iVideoFrameRate;
}ZBIT_VM_CONFIG, *PZBIT_VM_CONFIG;


typedef struct zbit_vm_frameinfo
{
	void *ptrY;
	void *ptrU;
	void *ptrV;
	int  m_iWidth;
	int  m_iHeight;
	int  m_iFrameRate;
	int  m_iFrameType;
	int  m_iSeq;
	int  m_iRes[64];
}ZBIT_VM_FRAMEINFO, *PZBIT_VM_FRAMEINFO;


class ZbitVideoMerge
{
public:
	ZbitVideoMerge(int ichannleID);
	virtual ~ZbitVideoMerge();

	int  init();

	int SetConfig(PZBIT_VM_CONFIG inputCfg);

	int VideoProcess(
		PZBIT_VM_FRAMEINFO IN frameA,
		PZBIT_VM_FRAMEINFO IN frameB,
		PZBIT_VM_FRAMEINFO IN frameC,
		PZBIT_VM_FRAMEINFO IN frameD,
		int IN inframeSize,
		PZBIT_VM_FRAMEINFO OUT framOUT);

public:
	int m_IchannleID;
	PZBIT_VM_CONFIG m_InputCfg;

	int iFirstTime;

	unsigned char *strinBufferA;
	unsigned char *strinBufferB;
	unsigned char *strinBufferC;
	unsigned char *strinBufferD;
	unsigned char *strOutBuffer;

	int iMergeWidth;/*��һ���ںϷ���������ͼ���*/
	int iMergeHight;/*��һ���ںϷ���������ͼ���*/

};


//opencv 2.4.8   Զ��  �������ֳ���ģʽ

/*====================================================================
    ������      :ZBIT_VM_init
    ����        :�㷨��ʼ��
    �������˵��:
    ����ֵ˵��  ��0 = �ɹ�������= ʧ��
******************************************************************************/
int ZBIT_VM_init();

/*====================================================================
    ������      :ZBIT_VM_create
    ����        :��������ͨ��
    �������˵��:
    ����ֵ˵��  ��NULL == ʧ��
******************************************************************************/
Video_Merge_Handler ZBIT_VM_create();


/*====================================================================
    ������      :ZBIT_VM_setConfig
    ����        :����ͨ��������
    �������˵��:
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
******************************************************************************/
int ZBIT_VM_setConfig(Video_Merge_Handler handid, PZBIT_VM_CONFIG inputCfg);

/*====================================================================
    ������      :ZBIT_VM_algProcess
    ����        :����������
    �������˵��:
    handid = �����ľ��
    frameA  frameB  frameC frameD ��Ӧ������Ĳ�ͬ��λ��YUV֡
    inframeSize = Ŀǰ����Դ�ĵ�λ��Ŀ
    framOUT = �ں�֮���һ֡YUV
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
******************************************************************************/
int ZBIT_VM_algProcess(Video_Merge_Handler handid, 
		PZBIT_VM_FRAMEINFO IN frameA, 
		PZBIT_VM_FRAMEINFO IN frameB, 
		PZBIT_VM_FRAMEINFO IN frameC, 
		PZBIT_VM_FRAMEINFO IN frameD, 
		int IN inframeSize,
		PZBIT_VM_FRAMEINFO OUT framOUT );

/*====================================================================
    ������      :ZBIT_VM_delete
    ����        :ɾ������ͨ��
    �������˵��:
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
******************************************************************************/
int ZBIT_VM_delete(Video_Merge_Handler handid);

/*====================================================================
    ������      :ZBIT_VM_uninit
    ����        :�㷨ģ�����Դ�ͷ�
    �������˵��:
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
******************************************************************************/
int ZBIT_VM_uninit();

/*====================================================================
    ������      :ZBIT_VM_getAlgVersion
    ����        : ��ȡ�㷨ģ��İ汾
    �������˵��:
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
******************************************************************************/
char * ZBIT_VM_getAlgVersion(char * verString, int len);
#ifdef __cplusplus
}
#endif // extern "C"

#endif
