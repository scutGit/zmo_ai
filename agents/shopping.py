"""
Shopping Agent — 搜索符合预算和场合的在线商品
真实版本: Google Shopping API / Amazon Product API
Mock 版本: 本地 mock 商品数据
"""
from __future__ import annotations
from models import ShoppingResult, UserIntent
import tools


class ShoppingAgent:

    async def search(self, intent: UserIntent) -> list[ShoppingResult]:
        query = f"{intent.occasion} 外套 商务 {intent.location}"
        print(f"\n  [Shopping] 搜索: '{query}' (预算 ≤ {intent.budget} {intent.currency})")

        results = await tools.search_products(
            query=query,
            max_price=intent.budget,
            category="jacket",
        )

        print(f"  [Shopping] ✓ 找到 {len(results)} 件商品:")
        for r in results:
            print(f"    - {r.product_name}: ${r.price} ★{r.rating}")

        return results
