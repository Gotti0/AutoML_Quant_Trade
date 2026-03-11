"""
AutoML_Quant_Trade - Google Cloud Storage 매니저

로컬 스토리지에 적재된 주식 데이터(SQLite DB, Parquet 등) 및 모델 파일을
GCP Cloud Storage(GCS) 버킷과 동기화(업로드/다운로드)하는 인터페이스.
"""
import os
import logging
from typing import Optional

try:
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
except ImportError:
    storage = None
    DefaultCredentialsError = Exception

from backend.config.settings import Settings

logger = logging.getLogger(__name__)


class GCSManager:
    """Google Cloud Storage 업로드/다운로드 관리자"""

    def __init__(self, bucket_name: str = "automl-quant-trade-bucket", project_id: Optional[str] = None):
        """
        초기화 시 GCS 클라이언트 인증을 시도합니다.
        GCP 환경(또는 로컬 GOOGLE_APPLICATION_CREDENTIALS 환경변수)이
        구성되어 있어야 정상 작동합니다.
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.client = None
        self.bucket = None

        if storage is None:
            logger.warning("google-cloud-storage 라이브러리가 설치되지 않았습니다. GCS 연동이 비활성화됩니다.")
            return

        try:
            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"Connected to GCS Bucket: {self.bucket_name}")
        except DefaultCredentialsError:
            logger.warning("GCP 인증 정보를 찾을 수 없습니다. (GOOGLE_APPLICATION_CREDENTIALS 확인)")
        except Exception as e:
            logger.error(f"GCS 초기화 중 알 수 없는 오류 발생: {e}")

    def is_enabled(self) -> bool:
        """클라이언트가 정상 연결되어 GCS 사용이 가능한지 반환"""
        return self.client is not None and self.bucket is not None

    def upload_file(self, source_file_name: str, destination_blob_name: str) -> bool:
        """
        로컬 파일을 GCS 버킷에 업로드.

        Parameters:
            source_file_name: 업로드할 로컬 파일의 절대 또는 상대 경로
            destination_blob_name: GCS 버킷 내 저장될 객체(Object) 이름
        Returns:
            성공 여부 (True/False)
        """
        if not self.is_enabled():
            logger.error("GCSManager가 활성화되지 않아 업로드를 수행할 수 없습니다.")
            return False

        if not os.path.exists(source_file_name):
            logger.error(f"업로드할 로컬 파일을 찾을 수 없습니다: {source_file_name}")
            return False

        try:
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_name)
            logger.info(f"파일 업로드 완료: {source_file_name} -> gs://{self.bucket_name}/{destination_blob_name}")
            return True
        except Exception as e:
            logger.error(f"업로드 실패 ({source_file_name} -> {destination_blob_name}): {e}")
            return False

    def download_file(self, source_blob_name: str, destination_file_name: str) -> bool:
        """
        GCS 버킷에서 로컬로 파일을 다운로드.

        Parameters:
            source_blob_name: GCS 버킷 내 저장된 객체(Object) 이름
            destination_file_name: 다운로드받을 로컬 파일 경로
        Returns:
            성공 여부 (True/False)
        """
        if not self.is_enabled():
            logger.error("GCSManager가 활성화되지 않아 다운로드를 수행할 수 없습니다.")
            return False

        try:
            blob = self.bucket.blob(source_blob_name)
            if not blob.exists():
                logger.error(f"GCS에 해당 파일이 존재하지 않습니다: {source_blob_name}")
                return False

            # 파일이 저장될 부모 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(os.path.abspath(destination_file_name)), exist_ok=True)
            
            blob.download_to_filename(destination_file_name)
            logger.info(f"파일 다운로드 완료: gs://{self.bucket_name}/{source_blob_name} -> {destination_file_name}")
            return True
        except Exception as e:
            logger.error(f"다운로드 실패 ({source_blob_name} -> {destination_file_name}): {e}")
            return False

    def sync_database_to_cloud(self) -> bool:
        """
        현재 로컬에 유지 중인 DB (quant_data.db)를 클라우드 버킷에 동기화.
        주요 사용처: 매일 데이터 수집 파이프라인(Phase 1) 종료 후 클라우드 학습 지원용
        """
        db_path = Settings.DB_PATH
        gcs_blob_name = "database/quant_data.db"
        return self.upload_file(db_path, gcs_blob_name)

    def sync_database_from_cloud(self) -> bool:
        """
        클라우드 버킷에 있는 최신 DB를 로컬로 다운로드.
        주요 사용처: 클라우드 스크리닝 서버나 신규 로컬 환경 세팅 시 최신 데이터 획득용
        """
        db_path = Settings.DB_PATH
        gcs_blob_name = "database/quant_data.db"
        return self.download_file(gcs_blob_name, db_path)

if __name__ == "__main__":
    # 라이브러리 및 환경변수 테스트용
    logging.basicConfig(level=logging.INFO)
    manager = GCSManager()
    if manager.is_enabled():
        print("GCS Manager is ready to use.")
    else:
        print("GCS Manager is NOT ready. Please check credentials or network.")
