#include"ZbitVideoMergeAlg.h"
#include <string>
#define CRTDBG_MAP_ALLOC  
#include <stdlib.h>  
#include <crtdbg.h> 
#include<array>
using namespace std;
//C++ͨ�����ַ��������һ���ַ��ĺ���洢һ���ַ�'\0'���������������ַ�����ĳ���
//str[i]=str[i]-'a'+'A';Сд���д   ����str[i]='3' str[i]-'0'��������3


//�����õݹ��ʱ�򣬿��԰�һЩ��Ҫ�Ķ�����Ϊ�ݹ麯�����������߷���ֵ����
//Ҫô�õݹ飬Ҫô�õ�����������ʱ�����ǿ��������Ǵ�ǰ�����ǴӺ���ǰ
//�ÿռ任ʱ�䣬�½�һ��vector����ż���ٰ����嵽ԭ����vector���棬�ܶ�ʱ��Ҫ�½�һ����������������һЩ�����������㷨�ϼ򻯺ܶ�
//�ݹ��ʱ����Щ��Ҫһֱ����ı�����Ҫȥ�ⲿ���壬
//ע����Խ����ڵݹ�����Ӽ��仰����


//int a=9/4����2�����Ҫ����3�Ļ����ǿ���dim3 grid((src1.cols + block.x - 1) / block.x, (src1.rows + block.y - 1) / block.y);��������cuda������趨�ÿ����������趨ÿ�����ж����߳�

//��һ�������leetcode��Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.
//https://www.nowcoder.com/practice/bd73f6b52fdc421d91b14f9c909f9104?tpId=46&tqId=29040&tPage=1&rp=1&ru=/ta/leetcode&qru=/ta/leetcode/question-ranking
//��̬�滮һ�����ڿ��Խ�������ָ��С���⣬���ҿ���ͨ��С����Ľ�����������⣬���������ַ���������һ����newһ��length+1�����飬Ȼ����㵽dp[0]��dp[n],ÿ�μ���dp[i]����һ�δ�i��n-1��0��i�ı���
//�ܶ�ʱ��Ҫ�½�һ����������������һЩ�����������㷨�ϼ򻯺ܶ�



//int array[3][5];
//
//int(*ch)[5] = array; //��ȷ������ָ���������ͬ�����൱��(int (*ch)[5] = &array[0];)
//int **ch = array; //�������ȷ����Ϊǰ����ָ������ָ���ָ�룬������ָ�����������ָ�룬���Ͳ�ͬ��������������ҲҪע�⡣


//string a=s.substr(0,5);       //����ַ���s�� �ӵ�0λ��ʼ�ĳ���Ϊ5���ַ���//Ĭ��ʱ�ĳ���Ϊ�ӿ�ʼλ�õ�β


//����ָ���ҵ������м�ڵ�
//ListNode* slow = head;
//ListNode* fast = head;
//while (fast->next != null&&fast->next->next != null){
//	fast = fast->next->next;
//	slow = slow->next;
//}
//��ʱslow�����м�ڵ�




//atoi(x.c_str());��string�͵�xת��int��   
//int a = 99;  char c[3] = "";  _itoa_s(a, c, 3, 10);���ǽ�int��תchar *

//string s;cin>>s;����string str(s.size(),'*');

//vector<int> a; vector<int> temp; temp=a; vector<int> temp2(b,e);����һ���������������������λ��[b,e)�е�������ָʾԪ�ص�һ������

//students.erase(students.begin()+i);ע�����Ҫ����ѭ�����棬���ж�����Ҫ��while(i!=students.size())����Ϊÿ��erase���޸�students.size()

//insert��erase������c.insert(d,b,e);��c.erase(b,e);����b,e���ǵ�����


//for (auto it = bottom.begin(); it != bottom.end(); it++)
//	ret.push_back(*it);
//�ȼ���ret.insert(ret.end(), bottom.begin(), bottom.end());


//sort(numbers.begin(), numbers.end()); // ����ȡ�����м��Ǹ���
//int middle = numbers[numbers.size() / 2];


//bool compare(const Student& x, const Student& y)
//{
//	return x.name < y.name;
//}
//
//sort(student.begin(),student.end(),compare);


//������������㷨��������˫������
//void convertHelper(TreeNode* cur, TreeNode*& pre)
//{
//	if (cur == nullptr) return;
//
//	convertHelper(cur->left, pre);
//
//	cur->left = pre;
//	if (pre) pre->right = cur;
//	pre = cur;
//
//	convertHelper(cur->right, pre);
//}


//���ݾ����ڵݹ���һ��ѵݹ�ǰ��һЩ�Ѹı������˻�
//class Solution {
//public:
//	vector<vector<int> > buffer;
//	vector<int> tmp;

//	vector<vector<int> > FindPath(TreeNode* root, int expectNumber) {
//		if (root == NULL)
//			return buffer;
//		tmp.push_back(root->val);
//		if ((expectNumber - root->val) == 0 && root->left == NULL && root->right == NULL)
//		{
//			buffer.push_back(tmp);
//		}
//		FindPath(root->left, expectNumber - root->val);
//		FindPath(root->right, expectNumber - root->val);
//
//		tmp.pop_back();
//		return buffer;
//	}
//};





//��student.a am I������������ʶ������һ�ԭ���Ѿ��ӵ��ʵ�˳��ת�ˣ���ȷ�ľ���Ӧ���ǡ�I am a student.��
//class Solution {
//public:
//	string ReverseSentence(string str) {
//		string res = "", tmp = "";
//		for (unsigned int i = 0; i < str.size(); ++i){
//			if (str[i] == ' ') res = " " + tmp + res, tmp = "";
//			else tmp += str[i];
//		}
//		if (tmp.size()) res = tmp + res;
//		return res;
//	}
//};




//����һ�����������飬����������������ƴ�������ų�һ��������ӡ��ƴ�ӳ���������������С��һ����������������{3��32��321}�����ӡ���������������ųɵ���С����Ϊ321323��


/*��vector�����ڵ����ݽ������򣬰��� ��a��bתΪstring��
�� a��b<b+a  a������ǰ �Ĺ�������,
�� 2 21 ��Ϊ 212 < 221 ���� �����Ϊ 21 2
to_string() ���Խ�int ת��Ϊstring
*/ 

//class Solution {
//public:
//	static bool cmp(int a, int b){
//		string A = "";
//		string B = "";
//		A += to_string(a);
//		A += to_string(b);
//		B += to_string(b);
//		B += to_string(a);
//
//		return A<B;
//	}
//	string PrintMinNumber(vector<int> numbers) {
//		string  answer = "";
//		sort(numbers.begin(), numbers.end(), cmp);
//		for (int i = 0; i<numbers.size(); i++){
//			answer += to_string(numbers[i]);
//		}
//		return answer;
//	}
//};



//��.h��class����������int* m_flags;   ��.cpp��ʵ��	int m_total = nubmers->length(); m_flags = new int[m_total]; ����д���ǳ���д�������ڲ�֪����������������


//*p=*q������ָ����ȶ�������ָ��ָ���������ȡ�һ����ַֻ�ܷ�һ��ָ�룬���Բ������ж��ٸ�ָ����ȣ���ֻ������Щָ��ָ����ͬһ������������������ʱҪ��memcpy��������ָ�����
//�������������Ӿ�getchar();
//()�����ȼ�����*������float *g()��ʾһ�����������ķ�������Ϊָ�򸡵�����ָ�룬����fp��һ������ָ�룬(*fp)()���Ե����������
//char cmd[128];gets(cmd); char c;c=getchar();
//ע��c���sleep()����

//����Base��virtual void print(){ cout << "Base"; }��������Derived��void print(){ cout << "Derived"; }��������Base *point = new Derived();point->print();�����ʾ����Derived���������virtual��ʾ����Base
// virtual void setWTA_K(int wta_k) = 0; ���麯�������=0������ʾһ�����⺯��ֻ���ṩ��һ���ɱ������͸�д�Ľӿڡ����ǣ�����������ͨ��������Ʊ����á�����Ǵ����⺯��
//��������̳У�һ�����������⺯�����౻������ʶ��Ϊ������ࡣ��ͼ����һ���������Ķ��������ᵼ�±���ʱ�̴���
// Query �����˴����⺯��
// ����, ����Ա���ܴ��������� Query �����
// ok: NameQuery �е� Query �Ӷ���
//Query *pq = new NameQuery("Nostromo");
// ����: new ���ʽ���� Query ����
//Query *pq2 = new Query;
//�������ֻ����Ϊ�Ӷ�������ں������������С�





//ֻҪ��.h��д�˵ķ�����.cpp�����Ҫʵ�֣������������ᱨ��


//int calendar[13][31];  int (*monthp)[31]; monthp=calendar;

//calendar[month][day]=0����*(*(calendar+month)+day)=0

//char *r;strcpy(r,s);strcat(r,t);�Ǵ�ģ�Ҫchar *r;r=malloc(strlen(s)+strlen(t)+1);strcpy(r,s);strcat(r,t);




//
//int main()
//{
//	//batchProcess();
//
//	FILE* pFileOut = fopen("stitch.yuv", "w+");
//
//	ZBIT_VM_init();
//
//	Video_Merge_Handler ptemp = NULL;//����Video_Merge_Handler��defineΪ��void *
//	ptemp = ZBIT_VM_create();//���԰�ZbitVideoMerge *����void *��void *�ܱ������������Ҫ���õ�ʱ�򣬰�void *ǿ��ת����ZbitVideoMerge *�Ϳ����ˡ�����˵��int *����ֱ�Ӹ���void *��void *�ٱ�int *Ҫǿ��ת��
//	if (NULL == ptemp)
//	{
//		printf("func[%s] LIne[%d] create failed ! \r\n", __FUNCTIONW__, __LINE__);
//		return -1;
//	}
//
//	PZBIT_VM_CONFIG inputCfg = new zbit_vm_cfg;//PZBIT_VM_CONFIG�Ǳ�typedef����һ���ṹ��ָ�룬������new �ṹ�����ķ�ʽ���ܸ����ڴ棬Ȼ���м�Ҫ��������Ҫ�õı�������ֵ����
//
//	inputCfg->m_iInputVideoHeight = 720;
//	inputCfg->m_iInputVideoWidth = 1280;
//	inputCfg->m_iVideoFrameRate = 25;
//
//	ZBIT_VM_setConfig(ptemp, inputCfg);
//
//
//#if 0
//	ZbitVideoMerge zb(1);
//
//	PZBIT_VM_CONFIG inputCfg = new zbit_vm_cfg;
//	inputCfg->m_iInputVideoHeight = 720;
//	inputCfg->m_iInputVideoWidth = 1280;
//	inputCfg->m_iVideoFrameRate = 25;
//
//	zb.SetConfig(inputCfg);
//#endif
//
//	FILE* pFileIn1 = fopen("result.yuv", "rb+");
//	FILE* pFileIn2 = fopen("result1.yuv", "rb+");
//
//
//	//�����⼸�д���������yuv��Ƶ��֡������all��
//	fseek(pFileIn1, 0L, SEEK_END);
//	long size = ftell(pFileIn1);
//	int all = size / (1280 * 720 * 1.5);
//	rewind(pFileIn1);
//
//	int bufLen = 1280 * 720 * 3 / 2;
//	unsigned char* pYuvBuf1 = new unsigned char[bufLen];
//	unsigned char* pYuvBuf2 = new unsigned char[bufLen];
//	unsigned char* pYuvBuf3 = new unsigned char[4000 * 1080 * 3 / 2];
//
//
//	fread(pYuvBuf1, bufLen*sizeof(unsigned char), 1, pFileIn1);
//	fread(pYuvBuf2, bufLen*sizeof(unsigned char), 1, pFileIn2);
//
//
//
//	PZBIT_VM_FRAMEINFO IN frameA = new zbit_vm_frameinfo;
//
//	frameA->ptrY = pYuvBuf1;
//	frameA->ptrU = pYuvBuf1 + 1280 * 720;
//	frameA->ptrV = pYuvBuf1 + 1280 * 720 + 1280 * 720 / 4;
//
//
//	PZBIT_VM_FRAMEINFO IN frameB = new zbit_vm_frameinfo;
//
//	frameB->ptrY = pYuvBuf2;
//	frameB->ptrU = pYuvBuf2 + 1280 * 720;
//	frameB->ptrV = pYuvBuf2 + 1280 * 720 + 1280 * 720 / 4;
//
//	PZBIT_VM_FRAMEINFO IN frameOUT = new zbit_vm_frameinfo;
//
//
//	frameOUT->ptrY = pYuvBuf3;		//��ʵ����pYuvBuf3ֻ�Ǹ� new unsigned char[4000*1080*3/2];���������ǿյģ�����Ҫ�����ȸ���frameOUT->ptrY�ȣ�
//	//�������memcpy(framOUT->ptrY, yuvImg.data, MergeFrameSize);�����,memcpy�������ǰ������ָ�룬�������ǳ��ȣ�����ָ�뿪ʼ�ĸó����ֽڵ����ݸ�һָ�뿪ʼ�ĸó���
//	frameOUT->ptrU = pYuvBuf3 + 4000 * 1080;
//	frameOUT->ptrV = pYuvBuf3 + 4000 * 1080 + 4000 * 1080 / 4;
//
//	ZBIT_VM_algProcess(ptemp, frameA, frameB, NULL, NULL, 2, frameOUT);
//
//	//zb.VideoProcess(frameA, frameB, NULL, NULL, 2, frameOUT);
//
//
//	for (int a = 0; a < 10; a++)
//	{
//	
//		for (int i = 0; i < all; i++)
//		{
//			int start1 = clock();
//
//			fread(pYuvBuf1, bufLen*sizeof(unsigned char), 1, pFileIn1);//fread������һ��������ָ�룬������char *���ڶ����Ƕ�ȡ���ֽ�������������Ҫ�����ٸ��ڶ����������ĸ���fopen���ص��ļ�ָ��
//			//�����fread���ļ�ָ��ͻ�ͣ���ⲻ��ص��ļ���ͷ�����ܺã�
//			fread(pYuvBuf2, bufLen*sizeof(unsigned char), 1, pFileIn2);
//
//
//
//			frameA->ptrY = pYuvBuf1;
//			frameA->ptrU = pYuvBuf1 + 1280 * 720;
//			frameA->ptrV = pYuvBuf1 + 1280 * 720 + 1280 * 720 / 4;
//
//
//			frameB->ptrY = pYuvBuf2;
//			frameB->ptrU = pYuvBuf2 + 1280 * 720;
//			frameB->ptrV = pYuvBuf2 + 1280 * 720 + 1280 * 720 / 4;
//
//			//zb.VideoProcess(frameA, frameB, NULL, NULL, 2, frameOUT);
//
//			ZBIT_VM_algProcess(ptemp, frameA, frameB, NULL, NULL, 2, frameOUT);
//
//
//
//
//			fwrite(frameOUT->ptrY, frameOUT->m_iWidth * frameOUT->m_iHeight, 1, pFileOut);
//			fwrite(frameOUT->ptrU, frameOUT->m_iWidth * frameOUT->m_iHeight / 4, 1, pFileOut);
//			fwrite(frameOUT->ptrV, frameOUT->m_iWidth * frameOUT->m_iHeight / 4, 1, pFileOut);
//
//			//memcpy(pYuvBufs, yuvImg.data, bufLens*sizeof(unsigned char));
//
//			int start2 = clock();
//			cout << "total: " << start2 - start1 << "width: " << frameOUT->m_iWidth << "height:" << frameOUT->m_iHeight << endl;
//
//		}
//
//		rewind(pFileIn1);
//		rewind(pFileIn2);
//	}
//
//	delete[] pYuvBuf1;
//	delete[] pYuvBuf2;
//	delete[] pYuvBuf3;
//	delete[] frameA;
//	delete[] frameB;
//	delete[] frameOUT;
//	ZBIT_VM_delete(ptemp);
//
//
//	fclose(pFileIn1);
//	fclose(pFileIn2);
//	fclose(pFileOut);
//
//	return 1;
//}
//
//




int main()
{
	//batchProcess();

	//FILE* pFileOut = fopen("stitch.yuv", "w+");

	ZBIT_VM_init();

	Video_Merge_Handler ptemp = NULL;//����Video_Merge_Handler��defineΪ��void *
	ptemp = ZBIT_VM_create();//���԰�ZbitVideoMerge *����void *��void *�ܱ������������Ҫ���õ�ʱ�򣬰�void *ǿ��ת����ZbitVideoMerge *�Ϳ����ˡ�����˵��int *����ֱ�Ӹ���void *��void *�ٱ�int *Ҫǿ��ת��
	if (NULL == ptemp)
	{
		printf("func[%s] LIne[%d] create failed ! \r\n", __FUNCTIONW__, __LINE__);
		return -1;
	}

	zbit_vm_cfg cfg = { 1280, 720, 1, 25 };

	PZBIT_VM_CONFIG inputCfg = &cfg ;

	ZBIT_VM_setConfig(ptemp, inputCfg);

	FILE* pFileIn1 = fopen("result.yuv", "rb+");
	FILE* pFileIn2 = fopen("result1.yuv", "rb+");


	//�����⼸�д���������yuv��Ƶ��֡������all��
	fseek(pFileIn1, 0L, SEEK_END);
	long size = ftell(pFileIn1);
	int all = size / (1280 * 720 * 1.5);
	rewind(pFileIn1);

	const int bufLen = 1280*720*3/2;
	const int bufLenMax = 2500 * 720 * 3 / 2;
	

	//�����⼸�б�����öѷ��䣬��������ջ���䲻��1280*720*3/2��ô��Ŀռ䣬�ܼ򵥣�д��char c[1280*720*3/2]�����в���
	unsigned char* pYuvBuf1 = new unsigned char[bufLen];
	unsigned char* pYuvBuf2 = new unsigned char[bufLen];
	unsigned char* pYuvBuf3 = new unsigned char[3000 * 720 * 3 / 2];


	fread(pYuvBuf1, bufLen*sizeof(unsigned char), 1, pFileIn1);
	fread(pYuvBuf2, bufLen*sizeof(unsigned char), 1, pFileIn2);




	zbit_vm_frameinfo framea = { pYuvBuf1, pYuvBuf1 + 1280 * 720, pYuvBuf1 + 1280 * 720 + 1280 * 720 / 4 };

	PZBIT_VM_FRAMEINFO IN frameA = &framea;

	//frameA->ptrY = pYuvBuf1;
	//frameA->ptrU = pYuvBuf1 + 1280 * 720;
	//frameA->ptrV = pYuvBuf1 + 1280 * 720 + 1280 * 720 / 4;


	zbit_vm_frameinfo frameb = { pYuvBuf2, pYuvBuf2 + 1280 * 720, pYuvBuf2 + 1280 * 720 + 1280 * 720 / 4 };

	PZBIT_VM_FRAMEINFO IN frameB = &frameb;

	//frameB->ptrY = pYuvBuf2;
	//frameB->ptrU = pYuvBuf2 + 1280 * 720;
	//frameB->ptrV = pYuvBuf2 + 1280 * 720 + 1280 * 720 / 4;

	zbit_vm_frameinfo frameout = { pYuvBuf3, pYuvBuf3 + 3000 * 720, pYuvBuf3 + 3000 * 720 + 3000 * 720 / 4 };

	PZBIT_VM_FRAMEINFO IN frameOUT = &frameout;


	//frameOUT->ptrY = pYuvBuf3;		//��ʵ����pYuvBuf3ֻ�Ǹ� new unsigned char[4000*1080*3/2];���������ǿյģ�����Ҫ�����ȸ���frameOUT->ptrY�ȣ�
	////�������memcpy(framOUT->ptrY, yuvImg.data, MergeFrameSize);�����,memcpy�������ǰ������ָ�룬�������ǳ��ȣ�����ָ�뿪ʼ�ĸó����ֽڵ����ݸ�һָ�뿪ʼ�ĸó���
	//frameOUT->ptrU = pYuvBuf3 + 4000 * 1080;
	//frameOUT->ptrV = pYuvBuf3 + 4000 * 1080 + 4000 * 1080 / 4;

	ZBIT_VM_algProcess(ptemp, frameA, frameB, NULL, NULL, 2, frameOUT);

	//zb.VideoProcess(frameA, frameB, NULL, NULL, 2, frameOUT);


	


		for (int i = 0; i < all; i++)
		{
			int start1 = clock();

			fread(pYuvBuf1, bufLen*sizeof(unsigned char), 1, pFileIn1);//fread������һ��������ָ�룬������char *���ڶ����Ƕ�ȡ���ֽ�������������Ҫ�����ٸ��ڶ����������ĸ���fopen���ص��ļ�ָ��
			//�����fread���ļ�ָ��ͻ�ͣ���ⲻ��ص��ļ���ͷ�����ܺã�
			fread(pYuvBuf2, bufLen*sizeof(unsigned char), 1, pFileIn2);



			frameA->ptrY = pYuvBuf1;
			frameA->ptrU = pYuvBuf1 + 1280 * 720;
			frameA->ptrV = pYuvBuf1 + 1280 * 720 + 1280 * 720 / 4;


			frameB->ptrY = pYuvBuf2;
			frameB->ptrU = pYuvBuf2 + 1280 * 720;
			frameB->ptrV = pYuvBuf2 + 1280 * 720 + 1280 * 720 / 4;

	

			ZBIT_VM_algProcess(ptemp, frameA, frameB, NULL, NULL, 2, frameOUT);

			/*fwrite(frameOUT->ptrY, frameOUT->m_iWidth * frameOUT->m_iHeight, 1, pFileOut);
			fwrite(frameOUT->ptrU, frameOUT->m_iWidth * frameOUT->m_iHeight / 4, 1, pFileOut);
			fwrite(frameOUT->ptrV, frameOUT->m_iWidth * frameOUT->m_iHeight / 4, 1, pFileOut);*/

			//memcpy(pYuvBufs, yuvImg.data, bufLens*sizeof(unsigned char));

			int start2 = clock();
			cout << "total: " << start2 - start1 << "width: " << frameOUT->m_iWidth << "height:" << frameOUT->m_iHeight << endl;

		}

		rewind(pFileIn1);
		rewind(pFileIn2);


	delete[] pYuvBuf1;
	delete[] pYuvBuf2;
	delete[] pYuvBuf3;
	ZBIT_VM_delete(ptemp);

	fclose(pFileIn1);
	fclose(pFileIn2);
	//fclose(pFileOut);

	return 1;
}


