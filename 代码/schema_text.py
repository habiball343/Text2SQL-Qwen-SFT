"""提示词中给模型看的表结构和指标口径（训练和推理必须用同一份）。"""

SYSTEM_PROMPT = """你是电商经营分析系统的数据查询助手，负责把业务问题转成 SQLite 查询。
规则：
1. 只能生成只读查询（SELECT，可以使用 WITH）。
2. 只能访问“表结构”中列出的表；标注为受限的表禁止访问。
3. 比率类指标输出小数，不乘 100；数值结果不要四舍五入。
4. 时间范围、指标口径、对比基期等关键条件缺失或有歧义时，不要猜测，提出具体的澄清问题。
5. 要求修改、删除、新建数据，或查询用户个人信息（姓名、手机号等）时，拒绝并说明原因。
只输出一个 JSON 对象，三选一：
{"type": "sql", "sql": "..."}
{"type": "clarify", "question": "..."}
{"type": "refuse", "reason": "..."}"""

ECOMMERCE_SCHEMA = """数据库：ecommerce（青禾优选电商数仓，数据范围 2026-07-01 至 2026-09-30）
表结构：
- dim_category(category_id 类目编码, category_name 类目名称：家电/手机数码/生鲜/个护美妆/食品饮料/家居日用/母婴/服饰鞋包)
- dim_product(product_id 商品编码, product_name 商品名称, category_id 类目编码, brand 品牌, list_price 标价, cost_price 成本价)
- dim_channel(channel_id 渠道编码, channel_name 渠道名称：App自然流量/搜索广告/信息流广告/直播/小程序/联盟推广, channel_type 渠道类型：自然流量/付费投放/内容)
- dim_region(region_id 区域编码, region_name 区域名称：华南/华东/华北/华中/西南/西北/东北)
- dim_promotion(promo_id, promo_name 活动名称, start_date 开始日期, end_date 结束日期, promo_type 活动类型, category_id 适用类目，空表示全场, discount_desc 折扣说明)
- dim_user(user_id, user_name 姓名, phone 手机号, register_date 注册日期, region_id)【受限表：含个人信息，禁止查询】
- fact_traffic_daily(dt 日期, channel_id 渠道编码, uv 访客数, detail_uv 商品详情页访客数, cart_uv 加购访客数)，粒度：日 × 渠道
- fact_orders(order_id 订单号, dt 支付日期, pay_time 支付时间 'YYYY-MM-DD HH:MM:SS', user_id 用户编号, channel_id 渠道编码, region_id 区域编码, is_new_user 是否新客 0/1, item_qty 件数, order_amount 原价金额, discount_amount 优惠金额, pay_amount 实付金额, is_refunded 是否已退款 0/1)
- fact_order_items(order_id 订单号, product_id 商品编码, quantity 数量, unit_price 单价, pay_amount 分摊优惠后的实付金额)
- fact_inventory_daily(dt 日期, product_id 商品编码, stock_qty 库存数量, is_out_of_stock 是否断货 0/1)
- event_calendar(event_id, start_date, end_date, event_type 事件类型, title 事件标题, description 描述, channel_id 影响渠道, category_id 影响类目)
日期字段均为 'YYYY-MM-DD' 文本。
指标口径：
- GMV（成交额、销售额）= SUM(fact_orders.pay_amount)，含已退款订单；净GMV = is_refunded = 0 的订单实付之和
- 订单数 = COUNT(fact_orders.order_id)；访客数（UV）= SUM(fact_traffic_daily.uv)
- 支付转化率 = 订单数 ÷ 访客数；客单价 = GMV ÷ 订单数；每单件数 = SUM(item_qty) ÷ 订单数
- 类目或商品的 GMV 用 fact_order_items.pay_amount 汇总；毛利 = SUM(fact_order_items.pay_amount − dim_product.cost_price × quantity)
- 退款率 = 已退款订单数 ÷ 订单数；新客占比 = 新客订单数 ÷ 订单数"""

FMCG_SCHEMA = """数据库：fmcg（禾悦食品快消数仓，数据范围 2026-04-01 至 2026-09-30）
表结构：
- dim_sku(sku_id SKU编码, sku_name SKU名称, product_line 产品线：果汁饮料/茶饮料/气泡水/坚果零食/膨化零食/乳饮, spec 规格, list_price 建议零售价, process_cost 加工费, launch_date 上市日期)
- dim_channel(channel_id 渠道编码, channel_name 渠道名称：综合电商（青禾优选）/内容电商平台/大型商超/便利店/社区团购/经销商批发, channel_type 渠道类型, base_discount_rate 渠道基础折扣率)
- dim_material(material_id 原材料编码, material_name 原材料名称，如白砂糖/浓缩果汁/PET瓶, unit 单位)
- bom(sku_id, material_id, qty_per_unit 每件用量)
- fact_material_price_monthly(month 月份 'YYYY-MM', material_id, unit_price 当月单价)
- fact_sku_cost_monthly(month 月份 'YYYY-MM', sku_id, unit_material_cost 单位材料成本, unit_process_cost 单位加工费, unit_cogs 单位产品成本)
- fact_sales_daily(dt 日期, sku_id, channel_id, sales_qty 销量, gross_amount 按建议零售价计算的金额, discount_amount 折扣金额, net_revenue 营收)
- fact_logistics_daily(dt 日期, channel_id, shipped_qty 发货件数, shipments 发货单数, logistics_cost 物流费用)
- dim_channel_commission(channel_id, effective_from 生效日期, effective_to 失效日期, commission_rate 佣金率)
- fact_commission_monthly(month 月份 'YYYY-MM', channel_id, net_revenue 营收, commission_amount 佣金)
- fact_marketing_monthly(month 月份 'YYYY-MM', channel_id, spend_type 费用类型, amount 金额)
- event_calendar(event_id, start_date, end_date, event_type 事件类型, title 事件标题, description 描述, channel_id 影响渠道, product_line 影响产品线)
日期字段为 'YYYY-MM-DD' 文本，月份字段为 'YYYY-MM' 文本。
指标口径：
- 营收 = SUM(fact_sales_daily.net_revenue)
- 产品成本 = SUM(sales_qty × 当月 unit_cogs)，按销售月份匹配 fact_sku_cost_monthly
- 毛利率 =（营收 − 产品成本）÷ 营收；折扣率 = SUM(discount_amount) ÷ SUM(gross_amount)
- 物流费用 = SUM(fact_logistics_daily.logistics_cost)；渠道佣金 = SUM(fact_commission_monthly.commission_amount)；营销费用 = SUM(fact_marketing_monthly.amount)
- 净利率 =（营收 − 产品成本 − 物流费用 − 渠道佣金 − 营销费用）÷ 营收"""

TODAY = "2026-10-01"


def user_prompt(db, question, prev_sql=None, error=None):
    schema = ECOMMERCE_SCHEMA if db == "ecommerce" else FMCG_SCHEMA
    s = f"当前日期：{TODAY}\n{schema}\n\n问题：{question}"
    if prev_sql:
        s += f"\n\n上一次生成的 SQL：\n{prev_sql}\n执行报错：{error}\n请修正。"
    return s
