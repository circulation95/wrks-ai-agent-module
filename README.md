# Works AI Agent Module (PoC)

**Works AI Agent Module**은 다양한 업무 보조 및 채팅 기능을 수행하는 AI 에이전트들을 모듈화하여 관리하는 프로젝트입니다.  
이 리포지토리는 Works AI 서비스의 **PoC(Proof of Concept)**를 위해 개발되었으며, Python과 Poetry를 기반으로 형상 관리 및 배포 환경이 구성되어 있습니다.

## 📌 프로젝트 소개 (Project Overview)

사용자의 업무 생산성을 높이기 위해 **일반 채팅, 문서 분석, 업무 자동화(PPT, 엑셀, 이메일 등)** 등 특화된 목적을 가진 AI 에이전트들을 제공합니다. LangChain/LangGraph 프레임워크를 활용하여 확장 가능한 구조로 설계되었습니다.

## 🛠 기술 스택 (Tech Stack)

* **Language:** Python 3.11+
* **Dependency Management:** [Poetry](https://python-poetry.org/)
* **Framework:** LangChain, LangGraph
* **LLM:** OpenAI (GPT-4o, GPT-4.1 nano 등)
* **UI/Interface:** Streamlit (예정)
* **Search:** Tavily Search API

## 🤖 제공되는 에이전트 (Agents)

| 에이전트 명 | 사용 도구 | 기능 설명 |
| :--- | :--- | :--- |
| **신중한 똑쟁이** | 이미지 생성, 웹 검색, 문서 검색, 코드 생성 | GPT-4o 기반. 대화, 코딩, 검색, 그림 생성/인식 등 복합 작업 수행 |
| **티키타카 장인** | 웹 검색, 이미지 생성 | GPT-4.1 nano 기반. 가볍고 빠른 일상 대화 및 간단 검색/생성 |
| **문서 파일 검토** | 문서 검색 | PDF/TXT 업로드 문서 요약 및 질의응답(Q&A) |
| **데이터 분석** | 없음 | 엑셀/CSV 업로드 후 데이터 분석 및 편집 |
| **키워드 검색** | 웹 검색 | 키워드 입력 시 핵심 정보와 최신 소식 정리 |
| **뉴스 검색** | 웹 검색 | 키워드 관련 최신 뉴스 정리 및 제공 |
| **특수문자 제안** | 없음 | 상황에 적절한 특수문자 추천 |
| **세금 전문가** | 없음 | 기초적인 세무 상담 및 세금 관련 질의응답 |

## 🚀 설치 및 실행 가이드 (Installation)

이 프로젝트는 **Poetry**를 사용하여 의존성을 관리합니다.
