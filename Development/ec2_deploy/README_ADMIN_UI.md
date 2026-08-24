# KDIC AWS 관리자 UI 통합본

기존 `Development/ec2_deploy` 구조를 유지하면서 최신 관리자 UI와 챗봇 런타임 연동 API를 추가한 버전입니다.

## 주요 기능

- Elasticsearch 인덱스·문서·청크 조회
- 검색 파라미터 초안 작성, A/B 평가, 운영값 반영
- 평가 데이터셋 XLSX/CSV 업로드 및 검색 지표 비교
- 신규 청크 추가·기존 청크 제외를 초안으로 관리
- 승인된 변경만 챗봇과 같은 런타임에 반영
- 작업 스냅샷 조회와 롤백
- HCX API 키 연결 확인 및 AWS 환경파일 교체
- 8시간 HttpOnly 관리자 세션

## 실행 구조

```text
브라우저 /admin
  -> kdic-admin-ui.html
  -> /api/admin-ui/*
  -> kdic_admin_extension_aws.py
  -> kdic_ec2_launcher.py가 로드한 동일 챗봇 파이프라인
  -> Elasticsearch / PostgreSQL / HCX API
```

관리자 화면에서 승인한 파라미터와 청크 변경은 별도의 모의 검색기가 아니라 현재 챗봇과 동일한 파이프라인 객체에 적용됩니다.

## 변경 파일

- `kdic_deploy_assets/kdic-admin-ui.html`: AWS에서 바로 제공하는 단일 HTML 번들
- `admin_ui_source/`: 번들 생성 전 HTML/CSS/Vanilla JS 원본
- `kdic_deploy_assets/kdic_admin_extension_aws.py`: 관리자 인증·평가·초안·반영·롤백 API
- `kdic_deploy_assets/kdic_ec2_launcher.py`: 기존 챗봇 실행 후 관리자 API 연결
- `kdic_deploy_assets/2026-08-23-kdic-fastapi-service.py`: 관리자 상태 표시 정합성 보완
- `deploy/kdic.env.example`: `KDIC_ENV_FILE` 설정 예시

## 보안 원칙

- `KDIC_ADMIN_TOKEN`, `HCX_API_KEY`의 실제 값은 Git에 저장하지 않습니다.
- 로그인 성공 시 관리자 토큰을 브라우저 저장소에 보관하지 않고 8시간짜리 HttpOnly 쿠키를 사용합니다.
- API 키 조회 API는 원문을 반환하지 않고 설정 여부와 fingerprint만 반환합니다.
- 실제 키는 서버의 `/opt/kdic/kdic.env`에 저장하며 파일 권한은 `600`을 사용합니다.
- PEM, `.env`, 실행 로그, 사용자 업로드 데이터는 커밋하지 않습니다.

## AWS 환경변수

```dotenv
HCX_API_KEY=실제_키
ELASTICSEARCH_URL=http://172.31.37.117:9200
KDIC_DATABASE_URL=postgresql://...
KDIC_ADMIN_TOKEN=관리자_토큰
KDIC_ENV_FILE=/opt/kdic/kdic.env
```

실제 값이 들어간 파일은 저장소에 올리지 않습니다.

## 배포 확인

```bash
sudo systemctl restart kdic-api.service
sudo systemctl status kdic-api.service --no-pager
curl -fsS http://127.0.0.1:8501/api/health
```

정상 상태에서는 `admin_mode`가 `STAGED_WRITE`로 표시됩니다.

## 현재 영속성 범위

- 챗봇 세션과 작업: PostgreSQL 저장
- Elasticsearch 검색 데이터: Elasticsearch 저장
- HCX API 키: AWS 환경파일 저장
- 관리자 화면의 미반영 초안·업로드 평가셋·일부 작업 스냅샷: 현재 프로세스 메모리 저장

따라서 서버 재시작 전에 중요한 초안은 반영하거나 별도로 내보내야 합니다. 다음 단계에서는 관리자 초안과 평가 실행 이력을 PostgreSQL 테이블에 연결하는 것이 권장됩니다.

## 롤백

배포 전 백업 파일로 UI와 Python 파일을 복원한 뒤 서비스를 재시작합니다. 운영 서버의 최신 백업 위치는 `/opt/kdic/runtime/latest-admin-backup.txt`에서 확인할 수 있습니다.
