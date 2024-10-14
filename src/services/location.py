import aiohttp
from src.models import LocationReverse


class Location:

    def __init__(self) -> None:
        self.host = "nominatim.openstreetmap.org"
        self.headers = {
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "referer": f"https://{self.host}/"
        }

    async def _request(self, method: str = "reverse", params: dict | None = None) -> dict | None:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(
                    f"https://{self.host}/{method}",
                    headers=self.headers,
                    params={**params, **{"format": "jsonv2"}}
            ) as resp:
                json = await resp.json()
                return json

    async def reverse(self, lat: float, lon: float) -> LocationReverse:
        response = await self._request("reverse", params={
            "lat": lat, "lon": lon
        })
        return LocationReverse(**response)
