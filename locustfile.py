from locust import HttpUser, task, between


class IrisAppUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def load_homepage(self):
        self.client.get("/")

    @task(2)
    def health_check(self):
        self.client.get("/_stcore/health")