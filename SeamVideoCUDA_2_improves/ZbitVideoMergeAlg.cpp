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
#include "ZbitVideoMergeAlg.h"


using namespace std;
using namespace cv;
using namespace cv::gpu;

#ifndef MEMORY_DELETE(x)
#define MEMORY_DELETE(x)  if(x) {delete x; x=NULL;}
#endif

#ifndef MEMORY_DELETE_EX(x)
#define MEMORY_DELETE_EX(x)  if(x){ delete[] x; x=NULL;}
#endif

#ifndef MEMORY_DELETE_FREE(x)
#define MEMORY_DELETE_FREE(x)  if(x){ free(x) ; x=NULL;}         //delete �ڴ�����
#endif

#ifndef     NULL
#define	    NULL    0
#endif

#define maxResolutionWith     4500
#define maxResolutionHigh     3000

#define ALG_VERSION "zbit & Dr-guo Team. Fuc[videomerge]  Ver[V1.0] "

#define BIG 1000000

FILE* pFileOut = fopen("stitch.yuv", "w+");



/*====================================================================
    ������      :ZBIT_VM_init
    ����        :�㷨��ʼ��
    �������˵��:
    ����ֵ˵��  ��0 = �ɹ�������= ʧ��
-------------------------------------------------------------------------------
�޸ļ�¼:
��  ��           �汾        �޸���      �޸�����
9/8/2017           1.0              ����         ����
******************************************************************************/
int ZBIT_VM_init()
{
	return 1;
}

/*====================================================================
    ������      :ZBIT_VM_create
    ����        :��������ͨ��
    �������˵��:
    ����ֵ˵��  ��NULL == ʧ��
-------------------------------------------------------------------------------
�޸ļ�¼:
��  ��           �汾        �޸���      �޸�����
9/8/2017           1.0              ����         ����
******************************************************************************/
Video_Merge_Handler ZBIT_VM_create()
{
    ZbitVideoMerge *prtVideoMerge = new ZbitVideoMerge(1);
    if(prtVideoMerge == NULL)
    {
        return NULL;
    }
	
    return (Video_Merge_Handler)prtVideoMerge;
}


/*====================================================================
    ������      :ZBIT_VM_setConfig
    ����        :����ͨ��������
    �������˵��:
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
-------------------------------------------------------------------------------
�޸ļ�¼:
��  ��           �汾        �޸���      �޸�����
9/8/2017           1.0              ����         ����
******************************************************************************/
int ZBIT_VM_setConfig(Video_Merge_Handler handid, PZBIT_VM_CONFIG inputCfg)
{
	int iRet = 0;
	
	if (NULL == handid)
	{
		return -1;
	}

	ZbitVideoMerge *prtVideoMerge = (ZbitVideoMerge *)handid;

	iRet = prtVideoMerge->SetConfig(inputCfg);
	if (0 <= iRet)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}


/*====================================================================
    ������      :ZBIT_VM_algProcess
    ����        :����������
    �������˵��:
    handid = �����ľ��
    frameA  frameB  frameC frameD ��Ӧ������Ĳ�ͬ��λ��YUV֡
    inframeSize = Ŀǰ����Դ�ĵ�λ��Ŀ
    framOUT = �ں�֮���һ֡YUV
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
-------------------------------------------------------------------------------
�޸ļ�¼:
��  ��           �汾        �޸���      �޸�����
9/8/2017           1.0              ����         ����
******************************************************************************/
int ZBIT_VM_algProcess(Video_Merge_Handler handid, 
		PZBIT_VM_FRAMEINFO IN frameA, 
		PZBIT_VM_FRAMEINFO IN frameB, 
		PZBIT_VM_FRAMEINFO IN frameC, 
		PZBIT_VM_FRAMEINFO IN frameD, 
		int IN inframeSize,
		PZBIT_VM_FRAMEINFO OUT framOUT )
{
	int iRet = 0;
	
	if (NULL == handid)
	{
		return -1;
	}

	ZbitVideoMerge *prtVideoMerge = (ZbitVideoMerge*)handid;

	iRet = prtVideoMerge->VideoProcess(
		frameA,
		frameB,
		frameC,
		frameD,
		inframeSize,
		framOUT);
	
	if (0 <= iRet)
	{
		return 0;
	}
	else
	{
		return -1;
	}
}


/*====================================================================
    ������      :ZBIT_VM_delete
    ����        :ɾ������ͨ��
    �������˵��:
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
-------------------------------------------------------------------------------
�޸ļ�¼:
��  ��           �汾        �޸���      �޸�����
9/8/2017           1.0              ����         ����
******************************************************************************/
int ZBIT_VM_delete(Video_Merge_Handler handid)
{
	int iRet = 0;
	
	if (NULL == handid)
	{
		return -1;
	}
	else
	{
		MEMORY_DELETE(handid);
		return iRet;
	}
}


/*====================================================================
    ������      :ZBIT_VM_uninit
    ����        :�㷨ģ�����Դ�ͷ�
    �������˵��:
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
-------------------------------------------------------------------------------
�޸ļ�¼:
��  ��           �汾        �޸���      �޸�����
9/8/2017           1.0              ����         ����
******************************************************************************/
int ZBIT_VM_uninit()
{
	int iRet = 0;
	return iRet;
}

/*====================================================================
    ������      :ZBIT_VM_uninit
    ����        :�㷨ģ�����Դ�ͷ�
    �������˵��:
    ����ֵ˵��  ��0 = ��ʾ�ɹ���0 >  ��ʾʧ��
-------------------------------------------------------------------------------
�޸ļ�¼:
��  ��           �汾        �޸���      �޸�����
9/8/2017           1.0              ����         ����
******************************************************************************/
char * ZBIT_VM_getAlgVersion(char * verString, int len)
{
	int ifd = 0;
	char szVerBuild[128] = {0};
	char startTime[128] = {0};
	sprintf(szVerBuild, "%s Time[%s %s] ", ALG_VERSION, __TIME__, __DATE__);

	memcpy(verString, szVerBuild, strlen(szVerBuild));

	return verString;
}


ZbitVideoMerge::ZbitVideoMerge(int ichannleID)
{
	m_IchannleID = ichannleID;
	strinBufferA = NULL;
	strinBufferB = NULL;
	strinBufferC = NULL;
	strinBufferD = NULL;
	
	iFirstTime = 0;
	iMergeWidth = 0;
	iMergeHight = 0;
}


ZbitVideoMerge::~ZbitVideoMerge()
{

}

int ZbitVideoMerge::init()
{
	return 1;
}

int ZbitVideoMerge::SetConfig(PZBIT_VM_CONFIG inputCfg)
{
	m_InputCfg = inputCfg;
	return 0;
}



int ZbitVideoMerge::VideoProcess(
	PZBIT_VM_FRAMEINFO IN frameA,
	PZBIT_VM_FRAMEINFO IN frameB,
	PZBIT_VM_FRAMEINFO IN frameC,
	PZBIT_VM_FRAMEINFO IN frameD,
	int IN inframeSize,
	PZBIT_VM_FRAMEINFO OUT framOUT)
{
	if (2 == inframeSize)
	{
		if ((NULL == frameA) \
			|| (NULL == frameB))
		{
			printf("func[%s] Line[%d] input arge error \r\n", __FUNCTIONW__, __LINE__);
			return -1;
		}
	}

	if (3 == inframeSize)
	{
		if ((NULL == frameA) \
			|| (NULL == frameB) \
			|| (NULL == frameC))
		{
			printf("func[%s] Line[%d] input arge error \r\n", __FUNCTIONW__, __LINE__);
			return -1;
		}
	}

	if (4 == inframeSize)
	{
		if ((NULL == frameA) \
			|| (NULL == frameB) \
			|| (NULL == frameC) \
			|| (NULL == frameD))
		{
			printf("func[%s] Line[%d] input arge error \r\n", __FUNCTIONW__, __LINE__);
			return -1;
		}
	}

	int iWidth = m_InputCfg->m_iInputVideoWidth;
	int iHeigh = m_InputCfg->m_iInputVideoHeight;
	int iFrameSize = iWidth * iHeigh * 3 / 2;

	int Ysize = iWidth*iHeigh;
	int UVsize = (iWidth / 2) * (iHeigh / 2);

	cv::Mat yuvImg1;
	cv::Mat rgbImg1;
	if (NULL != frameA)
	{
		yuvImg1.create(iHeigh * 3 / 2, iWidth, CV_8UC1);
		memcpy(yuvImg1.data, frameA->ptrY, Ysize);
		memcpy(yuvImg1.data + Ysize, frameA->ptrU, UVsize);
		memcpy(yuvImg1.data + Ysize + UVsize, frameA->ptrV, UVsize);

		cv::cvtColor(yuvImg1, rgbImg1, CV_YUV2BGR_I420);
	}


	cv::Mat yuvImg2;
	cv::Mat rgbImg2;
	if (NULL != frameB)
	{
		yuvImg2.create(iHeigh * 3 / 2, iWidth, CV_8UC1);
		memcpy(yuvImg2.data, frameB->ptrY, Ysize);
		memcpy(yuvImg2.data + Ysize, frameB->ptrU, UVsize);
		memcpy(yuvImg2.data + Ysize + UVsize, frameB->ptrV, UVsize);

		cv::cvtColor(yuvImg2, rgbImg2, CV_YUV2BGR_I420);
	}

	if (NULL != frameC)
	{

	}

	if (NULL != frameD)
	{

	}

	//printf("file[%s] func:%s line:%d \r\n", __FILE__, __func__, __LINE__);

	/*����жϵ�һ���ں�û�������������*/
	if (0 == iFirstTime)
	{
		printf("file[%s] func:%s line:%d \r\n", __FILE__, __FUNCTIONW__, __LINE__);
		/*��Ϊ���ֵ�һ���ںϣ�ֻ�ǻ�ȡ���ں�֮��һ֡�� �� �� �ߣ� ��Later �ںϺ�����Ӧ��Ҳ�ܻ�ȡ����͸�*/
		Mat MergeRes = imageStitchForVideo(rgbImg1, rgbImg2);

		printf("file[%s] func:%s line:%d \r\n", __FILE__, __FUNCTIONW__, __LINE__);
		/*�ں�֮��ĵ�һ֡ͼ�� */
		iMergeHight = MergeRes.rows;//��ȡ֡�߶�
		iMergeWidth = MergeRes.cols; //��ȡ֡��� 
		printf("func[%s] Line[%d] MergeResolution[%d * %d ] \r\n", __FUNCTIONW__, __LINE__, iMergeWidth, iMergeHight);
		framOUT->m_iWidth = iMergeWidth;
		framOUT->m_iHeight = iMergeHight;
		/*��һ���ں�֮�󣬺����Ͳ���Ҫ���ж���*/
		iFirstTime = 1;

		return 0;
	}

	//printf("file[%s] func:%s line:%d \r\n", __FILE__, __func__, __LINE__);

	/*��Ϊ���ֵ�һ���ںϣ�ֻ�ǻ�ȡ���ں�֮��һ֡�� �� �� �ߣ� ��Later �ںϺ�����Ӧ��Ҳ�ܻ�ȡ����͸�*/
	Mat MergeRes = imageStitchForVideoLater(rgbImg1, rgbImg2);

	

	//printf("file[%s] func:%s line:%d \r\n", __FILE__, __func__, __LINE__);
	/*�ں�֮��ĵ�һ֡ͼ�� */
	int MergeframeH = iMergeHight;//��ȡ֡�߶�
	int MergeframeW = iMergeWidth; //��ȡ֡��� 
	int MergeFrameSize = MergeframeH * MergeframeW;

	cv::Mat yuvImg;
	if (0 == (MergeframeW % 2))
	{
		cv::cvtColor(MergeRes, yuvImg, CV_BGR2YUV_I420);
	}
	else
	{
		cv::Mat yuvtemp;
		resize(MergeRes, yuvtemp, Size(MergeframeW + 1, MergeframeH), 0, 0, CV_INTER_LINEAR);
		framOUT->m_iWidth = MergeframeW + 1;
		MergeFrameSize = MergeframeH * (MergeframeW + 1);
		//printf("func[%s] Line[%d] MergeResolution[%d * %d ] \r\n", __func__, __LINE__, framOUT->m_iWidth, framOUT->m_iHeight);
		cv::cvtColor(yuvtemp, yuvImg, CV_BGR2YUV_I420);
	}

	if (NULL != framOUT)
	{
		/*��mat �ں�֮���YUV �ֱ𿽱������framOUT �� ������*/
		//framOUT->m_iWidth = MergeframeW;
		//framOUT->m_iHeight = MergeframeH;

		

		memcpy(framOUT->ptrY, yuvImg.data, MergeFrameSize);
		memcpy(framOUT->ptrU, yuvImg.data + MergeFrameSize, MergeFrameSize / 4);
		memcpy(framOUT->ptrV, yuvImg.data + MergeFrameSize + MergeFrameSize / 4, MergeFrameSize / 4);


		//memcpy(pYuvBufs, yuvImg.data, bufLens*sizeof(unsigned char));
		//fwrite(pYuvBufs, bufLens*sizeof(unsigned char), 1, pFileOut);

		//memcpy(pYuvBufs, framOUT->ptrY, bufLens*sizeof(unsigned char));
		//fwrite(pYuvBufs, bufLens*sizeof(unsigned char), 1, pFileOut);

		//memcpy(pYuvBufsUV, framOUT->ptrU, bufLensUV*sizeof(unsigned char));
		//fwrite(pYuvBufsUV, bufLensUV*sizeof(unsigned char), 1, pFileOut);

		//memcpy(pYuvBufsUV, framOUT->ptrV, bufLensUV*sizeof(unsigned char));
		//fwrite(pYuvBufsUV, bufLensUV*sizeof(unsigned char), 1, pFileOut);
		fwrite(framOUT->ptrY, framOUT->m_iWidth * framOUT->m_iHeight, 1, pFileOut);
		fwrite(framOUT->ptrU, framOUT->m_iWidth * framOUT->m_iHeight / 4, 1, pFileOut);
		fwrite(framOUT->ptrV, framOUT->m_iWidth * framOUT->m_iHeight / 4, 1, pFileOut); 


	}

	return 0;
}



